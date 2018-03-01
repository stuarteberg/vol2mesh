from io import BytesIO
from pathlib import Path
import numpy as np

def write_obj(vertices_zyx, faces, normals_zyx=None, output_file=None):
    """
    Generate an OBJ file from the given (binary) data and write it to the given byte stream or file path.
    
    If no output stream/path is given, return a bytes object containing the OBJ data.

    vertices_zyx: np.ndarray, shape=(N,3), dtype=float
    faces: np.ndarray, shape=(N,3), dtype=int
    normals_zyx: np.ndarray, shape=(N,3), dtype=float (Optional.)
    
    Note: Each 'face' consists of 3 indexes, which correspond to indexes in the vertices_zyx.
          The indexes should be 0-based. (They will be converted to 1-based in the OBJ)
    """
    if normals_zyx is None:
        normals_zyx = np.zeros((0,3), np.float32)

    need_close = True

    if output_file is None:
        mesh_bytestream = BytesIO()
    elif isinstance(output_file, (str, Path)):
        mesh_bytestream = open(output_file, 'wb')
    else:
        assert hasattr(output_file, 'write')
        mesh_bytestream = output_file
        need_close = False
    
    try:
        _write_obj(vertices_zyx, faces, normals_zyx, mesh_bytestream)
        if output_file is None:
            return mesh_bytestream.getvalue()
    finally:
        if need_close:
            mesh_bytestream.close()

def _write_obj(vertices_zyx, faces, normals_zyx, mesh_bytestream):
    """
    Given lists of vertices and faces, write them to the given stream in .obj format.
    
    vertices_zyx: np.ndarray, shape=(N,3), dtype=float
    faces: np.ndarray, shape=(N,3), dtype=int
    normals_zyx: np.ndarray, shape=(N,3), dtype=float
    
    Note: Each 'face' consists of 3 indexes, which correspond to indexes in the vertices_zyx.
          The indexes should be 0-based. (They will be converted to 1-based in the OBJ)
        
    Returns:
        BytesIO
    """
    if len(vertices_zyx) == 0:
        # Empty meshes result in no bytes
        return

    mesh_bytestream.write(b"# OBJ file\n")

    # OBJ format: XYZ order
    vertices_xyz = vertices_zyx[:, ::-1]
    normals_xyz = normals_zyx[:, ::-1]

    for (x,y,z) in vertices_xyz:
        mesh_bytestream.write(f"v {x:.7g} {y:.7g} {z:.7g}\n".encode('utf-8'))

    for (x,y,z) in normals_xyz:
        mesh_bytestream.write(f"vn {x:.7g} {y:.7g} {z:.7g}\n".encode('utf-8'))

    # OBJ format: Faces start at index 1 (not 0)
    for (v1, v2, v3) in faces+1:
        if len(normals_xyz) > 0:
            mesh_bytestream.write(f"f {v1}//{v1} {v2}//{v2} {v3}//{v3}\n".encode('utf-8'))
        else:
            mesh_bytestream.write(f"f {v1} {v2} {v3}\n".encode('utf-8'))
    

def read_obj(mesh_bytestream):
    """
    Read the OBJ file from the given file stream and return the vertexes/faces/normals as numpy arrays.
    
    Note:
        Many OBJ features, such as texture coordinates (vt lines), parameter space vertices (vp lines),
        "line elements", "materials", or faces with explicit vertex normal indices are not supported.
        (We don't support anything that requires a slash ('/') in any face definition.)
        
        f 20 30 40 # <-- OK
        f 20//1    # <-- NOT SUPPORTED
    
    Returns:
        vertices_zyx, faces, normals_zyx
        
        Note that the 'faces' indexes are 0-based
        (python conventions, not OBJ conventions, which start with 1)
    """
    need_close = False
    
    if isinstance(mesh_bytestream, bytes):
        mesh_bytestream = BytesIO(mesh_bytestream)
        need_close = True
    if isinstance(mesh_bytestream, (str, Path)):
        mesh_bytestream = open(mesh_bytestream, 'rb')
        need_close = True
    
    try:
        # For potentially huge meshes, many lists-of-lists is very inefficient with RAM
        # Therefore we keep flattened lists, and reshape them afterwards.
        vertices_xyz_flat = []
        faces_flat = []
        normals_xyz_flat = []
        
        for line in mesh_bytestream:
            if line.startswith(b'v '):
                vertices_xyz_flat.extend(map(float, line.split()[1:]))
            if line.startswith(b'f '):
                faces_flat.extend(map(int, line.split()[1:]))
            if line.startswith(b'vn '):
                normals_xyz_flat.extend(map(float, line.split()[1:]))
    
        if len(vertices_xyz_flat) % 3:
            raise RuntimeError("Unexpected format: total vertex count is not divisible by 3!")
        if len(faces_flat) % 3:
            raise RuntimeError("Unexpected format: face vertex count is not divisible by 3!")
        if len(normals_xyz_flat) % 3:
            raise RuntimeError("Unexpected format: normal components count is not divisible by 3!")
    
        vertices_xyz = np.array(vertices_xyz_flat, dtype=np.float32).reshape((-1,3))
        faces = np.array(faces_flat, dtype=np.uint32).reshape((-1,3))
        normals_xyz = np.array(normals_xyz_flat, dtype=np.float32).reshape((-1,3))
        
        vertices_zyx = vertices_xyz[:, ::-1]
        normals_zyx = normals_xyz[:, ::-1]
        
        # In OBJ, faces start at index 1 (not 0), but we want to use numpy conventions
        faces -= 1
    
        if len(faces) > 0 and faces.max() >= len(vertices_zyx):
            raise RuntimeError(f"Unexpected format: A face referenced vertex {faces.max()}, which is out-of-bounds for the vertex list.")

        return vertices_zyx, faces, normals_zyx
    finally:
        if need_close:
            mesh_bytestream.close()
