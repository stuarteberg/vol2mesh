"""
Fetch a single supervoxel from DVID (as a mask) and generate a mesh for it.
If the supervoxel's bounding box is larger than 1 Gvoxel,
the mask is downloaded at smaller scale is selected.

The resulting mesh can be saved to a file or uploaded to DVID (or both).

See --help for details.

Requirements:

    conda install -c flyem-forge neuclease vol2mesh libdvid-cpp

Example Usage:
    
    python sv_to_mesh.py -m 10e6 -s=3 -d=0.2 -o mesh-1224133018.obj emdata3:8900 7254 segmentation 1224133018
"""
import os
import sys
import logging
import argparse

import numpy as np

from neuclease import configure_default_logging
from neuclease.util import box_to_slicing, Timer
from neuclease.dvid import fetch_sparsevol_coarse, post_supervoxel

from libdvid import DVIDNodeService

from vol2mesh import Mesh

DEFAULT_MAX_BOUNDING_BOX_VOL = 1e9

logger = logging.getLogger(__name__)


def main():
    configure_default_logging()
    
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--smoothing-iterations', '-s', type=int, default=0)
    parser.add_argument('--decimation-fraction', '-d', type=float, default=1.0)

    parser.add_argument('--format', '-f', choices=['drc', 'obj'])
    parser.add_argument('--output-path', '-o',
                        help='Optional. Must end with .obj or .drc')
    parser.add_argument('--tarsupervoxels-instance', '-t', type=str,
                        help='Optional. The name of a tarsupervoxels instance to post the mesh to, e.g. "segmenation_sv_meshes".')

    parser.add_argument('--max-bounding-box-voxels', '-m', type=float, default=DEFAULT_MAX_BOUNDING_BOX_VOL,
                        help="Optional.  Attempt to ensure that the downlaoded mask's bounding box will not exceed this volume."
                             "  (A high scale is used if necessary.)")

    parser.add_argument('server')
    parser.add_argument('uuid')
    parser.add_argument('segmentation_instance')
    parser.add_argument('supervoxel_id', type=np.uint64)
    
    args = parser.parse_args()
    
    if not args.output_path and not args.tarsupervoxels_instance:
        sys.stderr.write("Nothing to do: You must specify either an output path or a tarsupervoxels instance\n")
        sys.exit(1)

    if args.output_path and args.format:
        if args.format != os.path.splitext(args.output_path)[1][1:]:
            sys.exit(f"Specified format ({args.format}) conflicts with output filename.")
            sys.exit(1)

    if args.output_path:
        args.format = os.path.splitext(args.output_path)[1][1:]
    elif not args.format:
        args.format = 'drc' # default

    # Fetch supervoxel mask and generate mesh    
    mesh = sv_to_mesh( args.server,
                       args.uuid,
                       args.segmentation_instance,
                       args.supervoxel_id,
                       args.smoothing_iterations,
                       args.decimation_fraction,
                       args.max_bounding_box_voxels)
    
    # Serialize to a buffer (either .obj or .drc)
    logger.info(f"Serializing to {args.format}")
    mesh_bytes = mesh.serialize(fmt=args.format)

    # Write to file
    if args.output_path:
        logger.info(f"Writing {args.output_path}")
        with open(args.output_path, 'wb') as f:
            f.write(mesh_bytes)

    # Send to DVID
    if args.tarsupervoxels_instance:
        logger.info(f"Posting to {args.server} / {args.uuid} / {args.tarsupervoxels_instance}")
        post_supervoxel(args.server, args.uuid, args.tarsupervoxels_instance, args.supervoxel_id, mesh_bytes)

    logger.info("DONE.")


def sv_to_mesh(server, uuid, instance, sv, smoothing_iterations=0, simplification_fraction=1.0, max_box_volume=DEFAULT_MAX_BOUNDING_BOX_VOL):
    """
    Download a mask for the given supervoxel and generate a mesh from it.
    If the mask bounding box would be large at scale 0, a smaller scale will be used.
    The returned mesh will always use scale-0 coordinates, though.
    """
    with Timer("Fetching supervoxel mask", logger):
        mask, scale, scaled_box = fetch_supervoxel_mask(server, uuid, instance, sv, max_box_volume)
        fullres_box = scaled_box * (2**scale)

    with Timer(f"Generating mesh from scale {scale}", logger):
        mesh = Mesh.from_binary_vol(mask, fullres_box)
    
    with Timer(f"Smoothing ({smoothing_iterations})", logger):
        mesh.laplacian_smooth(smoothing_iterations)
    
    # If we chose a scale other than 0, automatically reduce the
    # amount of decimation, since there will already be fewer vertices at lower resolution.
    simplification_fraction *= (2**scale)**2
    simplification_fraction = min(1.0, simplification_fraction)
    
    with Timer(f"Decimating ({simplification_fraction})", logger):
        mesh.simplify(simplification_fraction, in_memory=True)

    logger.info(f"Mesh has {len(mesh.vertices_zyx)} vertices and {len(mesh.faces)} faces")
    return mesh
    

def fetch_supervoxel_mask(server, uuid, instance, sv, max_box_volume):
    """
    Fetch a mask for the given supervoxel.
    The mask will be downloaded at a scale which is chosen such that the
    mask's bounding box will not exceed the given volume.
    """
    coarse_coords = fetch_sparsevol_coarse(server, uuid, instance, sv, supervoxels=True)
    
    # (Note: sparsevol-coarse is returned at scale 6)
    box = (2**6) * np.array([  coarse_coords.min(axis=0),
                             1+coarse_coords.max(axis=0)])
    shape = box[1] - box[0]
    scale = 0
    
    # Select a scale
    while np.prod(shape) > max_box_volume:
        scale += 1
        box //= 2
        shape = box[1] - box[0]
    
    # Fetch sparse masks
    ns = DVIDNodeService(server, uuid)
    block_coords, block_masks = ns.get_sparselabelmask(sv, instance, scale, supervoxels=True)
    
    fetched_box = np.array([   block_coords.min(axis=0),
                            64+block_coords.max(axis=0)])
    fetched_shape = fetched_box[1] - fetched_box[0]
    
    # Combine sparse masks into a single array
    full_mask = np.zeros(fetched_shape, dtype=bool)
    for coord, mask in zip(block_coords, block_masks):
        mask_box = np.array([coord, coord+64]) - fetched_box[0]
        full_mask[box_to_slicing(*mask_box)] = mask
    
    return full_mask, scale, fetched_box


if __name__ == "__main__":
    main()
