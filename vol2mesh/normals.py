"""
This file contains functions to compute face normals and vertex normals.

It contains two versions of each function, one based on numpy, and another based on numba.
It turns out face normals are faster to compute with plain numpy,
but vertex normals are faster to compute with numba, IFF you have already computed the face normals.
"""
import numpy as np

try:
    import numba
    _numba_available = True
except ImportError:
    _numba_available = False


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
        face_normals = compute_face_normals_numpy(vertices_zyx, faces, not weight_by_face_area)

    # numba is slightly faster for vertex normals, but not face normals
    if _numba_available:
        assert vertices_zyx.dtype == np.float32, \
            f"Our numba implementation requires float32 vertices, not {vertices_zyx.dtype}"
        return compute_vertex_normals_numba(vertices_zyx, faces, weight_by_face_area, face_normals)
    else:
        return compute_vertex_normals_numpy(vertices_zyx, faces, weight_by_face_area, face_normals)


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
    # numpy is faster than numba for face normals.
    # Always use numpy.
    return compute_face_normals_numpy(vertices_zyx, faces, normalize)


def compute_face_normals_numpy(vertices_zyx, faces, normalize=False):
    corners_0 = vertices_zyx[faces[:,0]]
    corners_1 = vertices_zyx[faces[:,1]]
    corners_2 = vertices_zyx[faces[:,2]]

    v1 = corners_1 - corners_0
    v2 = corners_2 - corners_1

    v_normal = np.cross(v2, v1) # This ordering is required for correct sign,
                                # since the handedness of the coordinate system is different for zyx vs xyz
    
    if normalize:
        magnitudes = np.linalg.norm(v_normal, axis=-1)
        v_normal[magnitudes != 0, :] /= magnitudes[magnitudes != 0, None]
        
    assert v_normal.shape == faces.shape
    return v_normal


def compute_vertex_normals_numpy(vertices_zyx, faces, weight_by_face_area=False, face_normals=None):
    if face_normals is None:
        face_normals = compute_face_normals_numpy(vertices_zyx, faces, not weight_by_face_area)

    vertex_normals = np.zeros(vertices_zyx.shape, np.float32)

    # Each vertex normal is the average of the normals from its N adjacent faces.
    # But an easier way to write this is to realize that each face normal contributes
    # to exactly three vertex normals.  So just sum up each face's contributions
    # to its neighboring vertex normals.
    np.add.at(vertex_normals, faces[:, 0], face_normals)
    np.add.at(vertex_normals, faces[:, 1], face_normals)
    np.add.at(vertex_normals, faces[:, 2], face_normals)

    magnitudes = np.linalg.norm(vertex_normals, axis=-1)
    nonzero_mags = magnitudes != 0
    vertex_normals[nonzero_mags, :] /= magnitudes[nonzero_mags, None]

    return vertex_normals

##
## numba implementations
##
if _numba_available:
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
    def norm_l2(v):
        """
        Same as np.linalg.norm for a single-vector input.
        
        By avoiding np.linalg.norm, we can support running on numpy
        installs that were not compiled with BLAS.
        (Admittedly, that's a rare scenario.)
        """
        return np.sqrt((v**2).sum())
    
    
    @numba.jit(nopython=True, cache=True)
    def compute_face_normals_numba(vertices_zyx, faces, normalize=False):
        face_normals = np.zeros(faces.shape, np.float32)
         
        for i in range(len(faces)):
            face = faces[i]
            corners = vertices_zyx[(face,)]
            v1 = corners[1] - corners[0]
            v2 = corners[2] - corners[0]
     
            v_normal = cross(v2, v1)    # This ordering is required for correct sign,
                                        # since the handedness of the coordinate system is different for zyx vs xyz
            if normalize:
                magnitude = norm_l2(v_normal)
                if magnitude != 0.0:
                    v_normal[:] /= magnitude
             
            face_normals[i] = v_normal
     
        return face_normals
    
    
    @numba.jit(nopython=True, cache=True)
    def compute_vertex_normals_numba(vertices_zyx, faces, weight_by_face_area=False, face_normals=None):
        if face_normals is None:
            face_normals = compute_face_normals_numba(vertices_zyx, faces, not weight_by_face_area)
    
        vertex_normals = np.zeros(vertices_zyx.shape, np.float32)
    
        # Each vertex normal is the average of the normals from its N adjacent faces.
        # But an easier way to write this is to realize that each face normal contributes
        # to exactly three vertex normals.  So just sum up each face's contributions
        # to its neighboring vertex normals.
        for i in range(len(faces)):
            face = faces[i]
            fn = face_normals[i]
            for vi in range(3):
                vertex_normals[face[vi],:] += fn
    
        for i in range(len(vertex_normals)):
            vn = vertex_normals[i]
            magnitude = norm_l2(vn)
            if magnitude != 0:
                vn[:] /= magnitude
        
        return vertex_normals
