import numpy as np
from skimage.measure import marching_cubes_lewiner

class Mesh:
    """
    A simple convenience class to hold the elements of a mesh.
    """
    def __init__(self, vertices_zyx, faces, normals_zyx=None, box=None):
        """
        vertices_zyx: ndarray (N,3), float
        faces: ndarray (M,3), integer (each element is an index referring to an element of vertices_zyx
        normals_zyx: ndarray (N,3), float
        box: ndarray (2,3) Overall bounding box of the mesh.
            (The bounding box information is not stored in mesh files like .obj and .drc,
            but it is useful to store it here for programmatic manipulation.)
        """
        self.vertices_zyx = vertices_zyx
        self.faces = faces
        self.normals_zyx = normals_zyx

        if normals_zyx is None:
            self.normals_zyx = np.zeros((0,3), dtype=np.int32)

        for a in (self.vertices_zyx, self.faces, self.normals_zyx):    
            assert a.ndim == 2 and a.shape[1] == 3, f"Input array has wrong shape: {a.shape}"

        if box is not None:
            self.box = np.asarray(box)
            assert self.box.shape == (2,3) 
        elif len(self.vertices_zyx) == 0:
                # No vertices. Choose a box with huge "negative shape",
                # so that it will have no effect when merged with other meshes.
                MAX_INT = np.iinfo(np.int32).max
                MIN_INT = np.iinfo(np.int32).min
                self.box = np.array([[MAX_INT, MAX_INT, MAX_INT],
                                     [MIN_INT, MIN_INT, MIN_INT]], dtype=np.int32)
        else:
            self.box = np.array( [ self.vertices_zyx.min(axis=0),
                                   np.ceil( self.vertices_zyx.max(axis=0) ) ] ).astype(np.int32)


def binary_vol_to_mesh(downsampled_volume_zyx, fullres_box_zyx=None, method='skimage'):
    """
    Run marching cubes on the given volume and return a Mesh object.

    Args:
        downsampled_volume_zyx:
            A binary volume, possibly at a downsampled resolution.
        fullres_box_zyx:
            The bounding-box inhabited by the given volume, in FULL-res coordinates.
        method:
            Which library to use for marching_cubes. For now the only choice is 'skimage'.
    """
    assert downsampled_volume_zyx.ndim == 3
    
    if fullres_box_zyx is None:
        fullres_box_zyx = np.array([(0,0,0), downsampled_volume_zyx.shape])
    
    try:
        if method == 'skimage':
            vertices_zyx, faces, normals_zyx, _values = marching_cubes_lewiner(downsampled_volume_zyx, 0.5, step_size=1)
        else:
            raise RuntimeError(f"Uknown method: {method}")
    except ValueError:
        if downsampled_volume_zyx.all():
            # Completely full boxes are 
            empty_vertices = np.zeros( (0, 3), dtype=np.float32 )
            empty_faces = np.zeros( (0, 3), dtype=np.uint32 )
            return Mesh(empty_vertices, empty_faces, box=fullres_box_zyx)
        else:
            raise

    # Infer the resolution of the downsampled volume
    resolution = (fullres_box_zyx[1] - fullres_box_zyx[0]) // downsampled_volume_zyx.shape
    
    # Upscale and translate the mesh into place
    vertices_zyx[:] *= resolution
    vertices_zyx[:] += fullres_box_zyx[0]
    
    return Mesh(vertices_zyx, faces, normals_zyx, fullres_box_zyx)


def concatenate_meshes(meshes):
    """
    Combine the given list of Mesh objects into a single Mesh object,
    renumbering the face vertexes as needed, and expanding the bounding box
    to encompass the union of the meshes.
    """
    vertex_counts = np.fromiter((len(mesh.vertices_zyx) for mesh in meshes), np.int64, len(meshes))
    normals_counts = np.fromiter((len(mesh.normals_zyx) for mesh in meshes), np.int64, len(meshes))
    face_counts = np.fromiter((len(mesh.faces) for mesh in meshes), np.int64, len(meshes))

    assert (vertex_counts == normals_counts).all() or not normals_counts.any(), \
        "Mesh normals do not correspond to vertexes.\n"\
        "(Either exclude all normals, more make sure they match the vertexes in every mesh.)"
    
    # vertexes and normals as simply concatenated
    concatenated_vertices = np.concatenate( [mesh.vertices_zyx for mesh in meshes] )
    concatenated_normals = np.concatenate( [mesh.normals_zyx for mesh in meshes] )

    # Faces need to be renumbered so that they refer to the correct vertexes in the combined list.
    concatenated_faces = np.ndarray((face_counts.sum(), 3), np.uint32)

    vertex_offsets = np.add.accumulate(vertex_counts[:-1])
    vertex_offsets = np.insert(vertex_offsets, 0, [0])

    face_offsets = np.add.accumulate(face_counts[:-1])
    face_offsets = np.insert(face_offsets, 0, [0])
    
    for faces, face_offset, vertex_offset in zip((mesh.faces for mesh in meshes), face_offsets, vertex_offsets):
        concatenated_faces[face_offset:face_offset+len(faces)] = faces + vertex_offset

    # bounding box is just the min/max of all bounding coordinates.
    all_boxes = np.stack([mesh.box for mesh in meshes])
    total_box = np.array( [ all_boxes[:,0,:].min(axis=0),
                            all_boxes[:,1,:].max(axis=0) ] )

    return Mesh( concatenated_vertices, concatenated_faces, concatenated_normals, total_box )

