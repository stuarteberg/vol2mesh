"""
Functions to read/write mesh data in the binary format that neuroglancer uses internally.

The binary layout is shown below.


NOTE:
    All vertex coordinates must be written in nanometer units.
    FlyEM meshes from DVID are usually stored at 8.0 nm resolution,
    so if you are converting a mesh from DVID to ngmesh, you should probably
    pre-multiply all vertex coordinates by 8.0 before calling write_ngmesh().
    Similarly, if you are reading an ngmesh and plan to store it in DVID
    (or display it alongside DVID grayscale data), you should probably
    divide its vertices by 8.0 immediately after calling read_ngmesh().

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

def read_ngmesh(f, mutable=False):
    """
    Read vertices and faces from the given binary file object,
    which is in ngmesh format as described above.

    Args:
        f:
            An open binary file object
        mutable:
            If True, return mutable arrays (requires an extra copy)
        
    Returns:
        (vertices_xyz, faces)
        where vertices_xyz is a 2D array (N,3), in XYZ order.
    """
    num_vertices = np.frombuffer(f.read(4), np.uint32)[0]
    vertices_xyz = np.frombuffer(f.read(int(3*4*num_vertices)), np.float32).reshape(-1, 3)
    faces = np.frombuffer(f.read(), np.uint32).reshape(-1, 3)
    
    if mutable:
        return vertices_xyz.copy(), faces.copy()
    else:
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
    Write the given vertices (verbatim) and faces to the given
    binary file object, which must already be open.
    """
    f_out.write( np.uint32(len(vertices_xyz)) )
    f_out.write( vertices_xyz.astype(np.float32, 'C', copy=False) )
    f_out.write( faces.astype(np.uint32, 'C', copy=False) )


def concatenate_ngmesh_files(paths, output_path):
    """
    Concatenate the ngmesh files into a single, combined file.

    The vertex coordinates are simply concatenated into one big list,
    but the vertex indices in the faces array need to be offset
    according to the vertexes' new positions in the final vertex list.
    
    Args: 
        paths:
            A list of file paths to .ngmesh files (format described above)
        output_path:
            Where to write the combined .ngmesh file
    """
    all_verts = []
    all_faces = []
    
    num_verts = 0
    for path in paths:
        with open(path, 'rb') as f:
            verts, faces = read_ngmesh(f, True)

        faces += num_verts
        num_verts += len(verts)
        
        all_verts.append(verts)
        all_faces.append(faces)

    final_verts = np.concatenate(all_verts)
    final_faces = np.concatenate(all_faces)
    
    write_ngmesh(final_verts, final_faces, output_path)
