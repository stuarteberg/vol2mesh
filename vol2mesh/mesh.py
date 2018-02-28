import tempfile
import subprocess
from io import BytesIO
from shutil import copyfileobj
import numpy as np
from skimage.measure import marching_cubes_lewiner

from .normals import compute_vertex_normals
from .obj_utils import write_obj, read_obj
from .io_utils import TemporaryNamedPipe, AutoDeleteDir

class Mesh:
    """
    A class to hold the elements of a mesh.
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


    @classmethod
    def from_binary_vol(cls, downsampled_volume_zyx, fullres_box_zyx=None, method='skimage'):
        """
        Alternate constructor.
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
        else:
            fullres_box_zyx = np.asarray(fullres_box_zyx)
        
        # Infer the resolution of the downsampled volume
        resolution = (fullres_box_zyx[1] - fullres_box_zyx[0]) // downsampled_volume_zyx.shape

        try:
            if method == 'skimage':
                padding = np.array([0,0,0])
                
                # Tiny volumes trigger a corner case in skimage, so we pad them with zeros.
                # This results in faces on all sides of the volume,
                # but it's not clear what else to do.
                if (np.array(downsampled_volume_zyx.shape) <= 2).any():
                    padding = np.array([2,2,2], dtype=int) - downsampled_volume_zyx.shape
                    padding = np.maximum([0,0,0], padding)
                    downsampled_volume_zyx = np.pad( downsampled_volume_zyx, tuple(zip(padding, padding)), 'constant' )

                vertices_zyx, faces, normals_zyx, _values = marching_cubes_lewiner(downsampled_volume_zyx, 0.5, step_size=1)
                
                # Skimage assumes that the coordinate origin is CENTERED inside pixel (0,0,0),
                # whereas we assume that the origin is the UPPER-LEFT corner of pixel (0,0,0).
                # Therefore, shift the results by a half-pixel.
                vertices_zyx += 0.5

                if padding.any():
                    vertices_zyx -= padding
            else:
                raise RuntimeError(f"Uknown method: {method}")
        except ValueError:
            if downsampled_volume_zyx.all():
                # Completely full boxes are not meshable -- they would be
                # open on all sides, leaving no vertices or faces.
                # Just return an empty mesh.
                empty_vertices = np.zeros( (0, 3), dtype=np.float32 )
                empty_faces = np.zeros( (0, 3), dtype=np.uint32 )
                return Mesh(empty_vertices, empty_faces, box=fullres_box_zyx)
            else:
                raise
    
        
        # Upscale and translate the mesh into place
        vertices_zyx[:] *= resolution
        vertices_zyx[:] += fullres_box_zyx[0]
        
        return Mesh(vertices_zyx, faces, normals_zyx, fullres_box_zyx)

    @classmethod
    def from_binary_blocks(cls, downsampled_binary_blocks, fullres_boxes_zyx=None, method='skimage'):
        """
        Alternate constructor.
        Compute a mesh for each of the given binary volumes
        (scaled and translated according to its associated box),
        and concatenate them.
        """
        meshes = []
        if fullres_boxes_zyx is None:
            fullres_boxes_zyx = [None]*len(downsampled_binary_blocks)

        for binary_vol, fullres_box_zyx in zip(downsampled_binary_blocks, fullres_boxes_zyx):
            mesh = cls.from_binary_vol(binary_vol, fullres_box_zyx, method)
            meshes.append(mesh)

        return concatenate_meshes(meshes)
        

    def simplify(self, fraction):
        """
        Simplify this mesh in-place, by the given fraction (of the original vertex count).
        """
        # The fq-mesh-simplify tool rejects inputs that are too small (if the decimated face count would be less than 4).
        # We have to check for this in advance because we can't gracefully handle the error.
        # https://github.com/neurolabusc/Fast-Quadric-Mesh-Simplification-Pascal-/blob/master/c_code/Main.cpp
        if fraction is None or fraction == 1.0 or (len(self.faces) * fraction <= 4):
            return self

        # Normals are about to get discarded and recomputed anyway,
        # so delete them now to save some RAM and serialization time.
        self.normals_zyx = None
        
        obj_bytes = write_obj(self.vertices_zyx, self.faces)
        bytes_stream = BytesIO(obj_bytes)

        simplify_input_pipe = TemporaryNamedPipe('input.obj')
        simplify_input_pipe.start_writing_stream(bytes_stream)
    
        simplify_output_pipe = TemporaryNamedPipe('output.obj')
    
        cmd = f'fq-mesh-simplify {simplify_input_pipe.path} {simplify_output_pipe.path} {fraction}'
        proc = subprocess.Popen(cmd, shell=True)
        mesh_stream = simplify_output_pipe.open_stream('rb')
        
        # The fq-mesh-simplify tool does not compute normals.  Compute them.
        self.vertices_zyx, self.faces, _empty_normals = read_obj(mesh_stream)
        mesh_stream.close()

        self.normals_zyx = compute_vertex_normals(self.vertices_zyx, self.faces)
        
        proc.wait(timeout=1.0)
        if proc.returncode != 0:
            raise RuntimeError(f"Child process returned an error code: {proc.returncode}.\n"
                               f"Command was: {cmd}")

    def serialize(self, output_format='obj'):
        """
        Serialize the mesh data in either .obj or .drc format, and return a bytes object.
        """
        assert output_format in ('obj', 'drc')
        obj_bytes = write_obj(self.vertices_zyx, self.faces, self.normals_zyx)

        if output_format == 'obj':
            return obj_bytes

        elif output_format == 'drc':
            # Sadly, draco is incapable of reading from non-seekable inputs.
            # It requires an actual input file, so we can't use a named pipe to avoid disk I/O.
            # But at least we can use a pipe for the output...
            mesh_dir = AutoDeleteDir(tempfile.mkdtemp())
            mesh_path = f'{mesh_dir}/mesh.obj'
            with open(mesh_path, 'wb') as mesh_file, BytesIO(obj_bytes) as obj_stream:
                copyfileobj(obj_stream, mesh_file)
            
            draco_output_pipe = TemporaryNamedPipe('output.drc')
            cmd = f'draco_encoder -cl 5 -i {mesh_path} -o {draco_output_pipe.path}'
            
            proc = subprocess.Popen(cmd, shell=True)
            with draco_output_pipe.open_stream('rb') as drc_stream:
                drc_bytes = drc_stream.read()

            proc.wait(timeout=600.0) # 10 minutes
            if proc.returncode != 0:
                raise RuntimeError(f"Child process returned an error code: {proc.returncode}.\n"
                                   f"Command was: {cmd}")

            return drc_bytes

def concatenate_meshes(meshes):
    """
    Combine the given list of Mesh objects into a single Mesh object,
    renumbering the face vertices as needed, and expanding the bounding box
    to encompass the union of the meshes.
    """
    vertex_counts = np.fromiter((len(mesh.vertices_zyx) for mesh in meshes), np.int64, len(meshes))
    normals_counts = np.fromiter((len(mesh.normals_zyx) for mesh in meshes), np.int64, len(meshes))
    face_counts = np.fromiter((len(mesh.faces) for mesh in meshes), np.int64, len(meshes))

    assert (vertex_counts == normals_counts).all() or not normals_counts.any(), \
        "Mesh normals do not correspond to vertices.\n"\
        "(Either exclude all normals, more make sure they match the vertices in every mesh.)"
    
    # vertices and normals as simply concatenated
    concatenated_vertices = np.concatenate( [mesh.vertices_zyx for mesh in meshes] )
    concatenated_normals = np.concatenate( [mesh.normals_zyx for mesh in meshes] )

    # Faces need to be renumbered so that they refer to the correct vertices in the combined list.
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

