from io import BytesIO
from pathlib import Path
import numpy as np

def write_obj(vertices_xyz, faces, normals_xyz=None, output_file=None):
    """
    Generate an OBJ file from the given (binary) data and write it to the given byte stream or file path.
    
    If no output stream/path is given, return a bytes object containing the OBJ data.

    vertices_xyz: np.ndarray, shape=(N,3), dtype=float
    faces: np.ndarray, shape=(N,3), dtype=int
    normals_xyz: np.ndarray, shape=(N,3), dtype=float (Optional.)
    
    Note: Each 'face' consists of 3 indexes, which correspond to indexes in the vertices_xyz.
          The indexes should be 0-based. (They will be converted to 1-based in the OBJ)
    """
    if normals_xyz is None:
        normals_xyz = np.zeros((0,3), np.float32)

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
        _write_obj(vertices_xyz, faces, normals_xyz, mesh_bytestream)
        if output_file is None:
            return mesh_bytestream.getvalue()
    finally:
        if need_close:
            mesh_bytestream.close()

def _write_obj(vertices_xyz, faces, normals_xyz, mesh_bytestream):
    """
    Given lists of vertices and faces, write them to the given stream in .obj format.
    
    vertices_xyz: np.ndarray, shape=(N,3), dtype=float
    faces: np.ndarray, shape=(N,3), dtype=int
    normals_xyz: np.ndarray, shape=(N,3), dtype=float
    
    Note: Each 'face' consists of 3 indexes, which correspond to indexes in the vertices_xyz.
          The indexes should be 0-based. (They will be converted to 1-based in the OBJ)
        
    Returns:
        BytesIO
    """
    if len(vertices_xyz) == 0:
        # Empty meshes result in no bytes
        return

    mesh_bytestream.write(b"# OBJ file\n")

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
        "line elements", "materials", or faces with out-of-order vertex normal indices are not supported.
        
        f 20 30 40                 # <-- OK
        f 20//20 30//30 40//40     # <-- OK
        f 20/1/20 30/2/30 40/3/40  # <-- OK, but texture coordinates will be discarded
        f 20//1 30//2 40//3        # <-- NOT SUPPORTED (out-of-order vertex normals)
    
    Returns:
        vertices_xyz, faces, normals_xyz
        
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
        faces_normal_indices_flat = []
        listed_normals_xyz_flat = []

        
        for line in mesh_bytestream:
            if line.startswith(b'v '):
                vertices_xyz_flat.extend(map(float, line.split()[1:]))
            if line.startswith(b'f '):
                for word in line.split()[1:]:
                    fields = word.split(b'/')
                    faces_flat.append(int(fields[0]))
                    if len(fields) == 3:
                        faces_normal_indices_flat.append(int(fields[2]))
            if line.startswith(b'vn '):
                listed_normals_xyz_flat.extend(map(float, line.split()[1:]))
    
        if len(vertices_xyz_flat) % 3:
            raise RuntimeError("Unexpected format: total vertex count is not divisible by 3!")
        if len(faces_flat) % 3:
            raise RuntimeError("Unexpected format: face vertex count is not divisible by 3!")
        if len(listed_normals_xyz_flat) % 3:
            raise RuntimeError("Unexpected format: normal components count is not divisible by 3!")
    
        vertices_xyz = np.array(vertices_xyz_flat, dtype=np.float32).reshape((-1,3))
        faces = np.array(faces_flat, dtype=np.uint32).reshape((-1,3))
        faces_normal_indices = np.array(faces_normal_indices_flat, dtype=np.uint32).reshape((-1,3))
        listed_normals_xyz = np.array(listed_normals_xyz_flat, dtype=np.float32).reshape((-1,3))

        # In OBJ, indices start at index 1 (not 0), but we want to use numpy conventions
        faces -= 1
        faces_normal_indices -= 1
        
        if len(faces_normal_indices) > 0:
            #
            # Notice that we don't permit the same vertex to have two different normals,
            # even if the vertex is referenced in two different faces.
            # Hence the caveat above about out-of-order vertex normals being unsupported..
            #
            normals_xyz = np.zeros(vertices_xyz.shape, dtype=np.float32)
            #
            # TODO:
            #   Speed up this loop with fancy indexing
            #   I think this would work (untested):
            #
            #     normals_xyz[(faces[:, 0],)] = listed_normals_xyz[(faces_normal_indices[:,0],)]
            #     normals_xyz[(faces[:, 1],)] = listed_normals_xyz[(faces_normal_indices[:,1],)]
            #     normals_xyz[(faces[:, 2],)] = listed_normals_xyz[(faces_normal_indices[:,2],)]
            #
            #
            for face, normal_indices in zip(faces, faces_normal_indices):
                normals_xyz[face[0]] = listed_normals_xyz[normal_indices[0]]
                normals_xyz[face[1]] = listed_normals_xyz[normal_indices[1]]
                normals_xyz[face[2]] = listed_normals_xyz[normal_indices[2]]

        elif len(listed_normals_xyz) > 0:
            if len(listed_normals_xyz) != len(vertices_xyz):
                raise RuntimeError("Listed normals do not match number of listed vertices")
        else:
            normals_xyz = np.zeros( (0,3), np.float32 )

       
        if len(faces) > 0 and faces.max() >= len(vertices_xyz):
            raise RuntimeError(f"Unexpected format: A face referenced vertex {faces.max()}, which is out-of-bounds for the vertex list.")

        return vertices_xyz, faces, normals_xyz
    finally:
        if need_close:
            mesh_bytestream.close()
