"""
Functions to read/write mesh data in the binary format that neuroglancer uses internally.

The binary format is simple:

num_vertices <uint32>
vertex <float32>,<float32>,<float32>
vertex <float32>,<float32>,<float32>
...
vertex <float32>,<float32>,<float32>
face <uint32>,<uint32>,<uint32>
face <uint32>,<uint32>,<uint32>
...
face <uint32>,<uint32>,<uint32>
<end-of-file>
"""
import io
import numpy as np

def read_ngmesh(f):
    """
    Read vertices and faces from the given binary file object,
    which is in ngmesh format as described above. 
    """
    num_vertices = np.frombuffer(f.read(4), np.uint32)[0]
    vertices_xyz = np.frombuffer(f.read(int(3*4*num_vertices)), np.float32).reshape(-1, 3)
    faces = np.frombuffer(f.read(), np.uint32).reshape(-1, 3)
    return vertices_xyz, faces

def write_ngmesh(vertices_xyz, faces, f_out=None):
    """
    Write the given vertices and faces to the given output path/file,
    in ngmesh format as described above.
    
    Args:
        vertices_xyz:
            vertex array, shape (V,3)
        
        faces:
            face index array, shape (F,3), referring to the rows of vertices_xyz
    
        f_out:
            If None, bytes are returned
            If a file path, the data is written to a file at that path.
            Otherwise, must be an open binary file object.
    Returns:
        If f_out is None, bytes are returned.
        Otherwise the mesh is written to f_out and None is returned.
    """
    if f_out is None:
        # Return as bytes
        with io.BytesIO() as bio:
            _write_ngmesh(vertices_xyz, faces, bio)
            return bio.getvalue()

    elif isinstance(f_out, str):
        # Write to a path
        with open(f_out, 'wb') as f:
            _write_ngmesh(vertices_xyz, faces, f)
    
    else:
        # Write to the given file object
        _write_ngmesh(vertices_xyz, faces, f)

def _write_ngmesh(vertices_xyz, faces, f_out):
    """
    Write the given vertices and faces to the given
    binary file object, which must already be open.
    """
    f_out.write(np.uint32(len(vertices_xyz)))
    f_out.write(vertices_xyz.astype(np.float32, 'C', copy=False))
    f_out.write(faces.astype(np.uint32, 'C', copy=False))
