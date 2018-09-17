"""
Download a supervoxel mesh tarfile from DVID and
concatenate the supervoxel meshes into a single mesh file.

Note: Requires neuclease (conda install -c flyem-forge neuclease)

Examples:
    mesh_from_dvid_tarfile emdata3:8900 0716 segmentation_sv_meshes 1668443473
    mesh_from_dvid_tarfile -s 0.5 -o '{body}-simplified.drc' emdata3:8900 0716 segmentation_sv_meshes 1668443473
"""
import logging
import argparse
from vol2mesh import Mesh

logger = logging.getLogger(__name__)


def main():
    from neuclease import configure_default_logging

    configure_default_logging()
    
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output-path', '-o', default='{body}.obj',
                        help='Output path.  If processing multiple bodies, use {body} in the name, e.g. "{body}.obj"')
    parser.add_argument('--simplify', '-s', type=float, default=1.0,
                        help='Optional decimation to apply before serialization, between 0.01 (most aggressive) and 1.0 (no decimation).')
    parser.add_argument('server')
    parser.add_argument('uuid')
    parser.add_argument('tarsupervoxels_instance')
    parser.add_argument('body', nargs='+')
    args = parser.parse_args()

    mesh_from_dvid_tarfile(args.server, args.uuid, args.tarsupervoxels_instance, args.body, args.simplify, args.output_path)
    logger.info("DONE")


def mesh_from_dvid_tarfile(server, uuid, tsv_instance, bodies, simplify=1.0, output_path='{body}.obj'):
    from neuclease.dvid import fetch_tarfile

    for body in bodies:
        logger.info(f"Body {body}: Fetching tarfile")
        tar_bytes = fetch_tarfile(server, uuid, tsv_instance, body)
        mesh = Mesh.from_tarfile(tar_bytes)
        mesh.simplify(simplify, in_memory=True)
        mesh.serialize(output_path.format(body=body))

if __name__ == "__main__":
    main()
