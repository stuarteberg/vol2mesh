import numpy as np
from .mesh import Mesh

def mesh_from_array(volume_zyx, global_offset_zyx, downsample_factor=1, simplify_ratio=None, output_format='obj', return_vertex_count=False):
    """
    Given a binary volume, convert it to a mesh in .obj format, optionally simplified.
    
    Parameters
    ----------
    
    volume_zyx:
        Binary volume (ZYX order)
    global_offset_zyx:
        Offset of the volume start corner in global non-downsampled coordinates: (z0,y0,x0)
    downsample_factor:
        Factor by which the given volume has been downsampled from its original size
    simplify_ratio:
        How much to simplify the generated mesh (or None to skip simplification)
    output_format:
        Either 'drc' or 'obj'
    return_vertex_count:
        If True, also return the APPROXIMATE vertex count
        (We don't count the vertexes after decimation; we assume that decimation
        was able to faithfully apply the requested simplify_ratio.)
    """
    assert output_format in ('obj', 'drc'), \
        f"Unknown output format: {output_format}.  Expected one of ('obj', 'drc')"

    box = [ global_offset_zyx,
            global_offset_zyx + downsample_factor * np.asarray(volume_zyx.shape) ]

    mesh = Mesh.from_binary_vol(volume_zyx, box, 'skimage')
    mesh.simplify( simplify_ratio )
    serialized_bytes = mesh.serialize(output_format)
    if return_vertex_count:
        return serialized_bytes, len(mesh.vertices_zyx)
    else:
        return serialized_bytes


