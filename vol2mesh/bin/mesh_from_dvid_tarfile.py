"""
Downloads one or more supervoxel mesh tarfiles from DVID.
For each tarfile, concatenates its supervoxel meshes into
a single mesh and writes it to disk.

Note: Requires neuclease (conda install -c flyem-forge neuclease)

Examples:
    # Three bodies
    mesh_from_dvid_tarfile emdata3:8900 0716 segmentation_sv_meshes 1640922516 1668443473 705722260
    
    # One body, decimated
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
                        help='Output path.  If processing multiple bodies, use {body} in the name. Default: "{body}.obj"')
    parser.add_argument('--simplify', '-s', type=float, default=1.0,
                        help='Optional decimation to apply before serialization, between 0.01 (most aggressive) and 1.0 (no decimation, the default).')
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

        logger.info(f"Body {body}: Loading mesh")
        mesh = Mesh.from_tarfile(tar_bytes)

        if simplify != 1.0:
            logger.info(f"Body {body}: Simplifying")
            mesh.simplify(simplify, in_memory=True)

        p = output_path.format(body=body)
        logger.info(f"Body {body}: Serializing to {p}")
        mesh.serialize(p)

if __name__ == "__main__":
    main()
