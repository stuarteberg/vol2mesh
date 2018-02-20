import numpy as np
import numba

@numba.jit(nopython=True, cache=True)
def cross(u,v):
    """
    numba doesn't support np.cross() out-of-the-box,
    so here it is.
    """
    u1, u2, u3 = u
    v1, v2, v3 = v
    return np.array([u2*v3 - u3*v2,
                     u3*v1 - u1*v3,
                     u1*v2 - u2*v1], dtype=u.dtype)

@numba.jit(nopython=True, cache=True)
def compute_face_normals(vertices_zyx, faces, normalize=False):
    """
    Compute the normal vector for the given triangular faces.
    
    The faces are specified in the typical fashion,
    i.e. each face's corners are specified as a list of 3 indices,
    indicating which vertices in the given vertex list comprise the face corners.
    
    If normalize=True, then unit vectors are returned.
    Otherwise, the magnitudes will be proportional to the areas of the faces.
    
    Faces with zero width will be given a normal of [0.0, 0.0, 0.0], regardless of the 'normalize' setting.
    """
    face_normals = np.zeros(faces.shape, np.float32)
    
    for i in range(len(faces)):
        face = faces[i]
        corners = vertices_zyx[(face,)]
        v1 = corners[1] - corners[0]
        v2 = corners[2] - corners[0]

        v_normal = cross(v2, v1)    # This ordering is required for correct sign,
                                    # since the handedness of the coordinate system is different for zyx vs xzy
        if normalize:
            norm = np.linalg.norm(v_normal)
            if norm != 0.0:
                v_normal[:] /= norm
        
        face_normals[i] = v_normal

    return face_normals

@numba.jit(nopython=True, cache=True)
def compute_vertex_normals(vertices_zyx, faces, weight_by_face_area=False, face_normals=None):
    """
    Compute the normal vector for each of the given vertexes
    in the mesh specified by the given vertexes and faces.

    Each vertex's normal vector is simply the average of the
    normal vectors of the faces it is adjacent to.

    Implementation ported from NeuTu:
    https://github.com/janelia-flyem/NeuTu/blob/7ffb7a/neurolabi/gui/zmesh.cpp#L424-L454

    Args:
        vertices_zyx: Numpy array shape=(N,3), dtype=float
        
        faces: Numpy array (M,3), dtype=integer
            The faces are specified in the typical fashion,
            i.e. each face's corners are specified as a list of 3 indices,
            indicating which vertices in the given vertex list comprise the face corners.
        
        weight_by_face_area:
            If True, larger faces will contribute proportionally
            more to their adjacent vertices' normals.
        
        face_normals: (Optional.) Numpy array shape=(M,3), dtype=float
            Pre-computed face normals, if you've got them handy.
            If not provided, they'll be computed first.
    
    Returns: Numpy array (N,3)
    """
    if face_normals is None:
        face_normals = compute_face_normals(vertices_zyx, faces, not weight_by_face_area)

    vertex_normals = np.zeros(vertices_zyx.shape, np.float32)

    for i in range(len(faces)):
        face = faces[i]
        fn = face_normals[i]
        for vi in range(3):
            vertex_normals[face[vi],:] += fn

    for i in range(len(vertex_normals)):
        vn = vertex_normals[i]
        norm = np.linalg.norm(vn)
        if norm != 0:
            vn[:] /= norm
    
    return vertex_normals
