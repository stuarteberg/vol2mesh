import os
import tempfile
import logging
import subprocess
from io import BytesIO
from shutil import copyfileobj

import numpy as np
import pandas as pd
from skimage.measure import marching_cubes_lewiner

from .normals import compute_vertex_normals
from .obj_utils import write_obj, read_obj
from .io_utils import TemporaryNamedPipe, AutoDeleteDir

logger = logging.getLogger(__name__)

DEBUG_DRACO = False

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
        self.vertices_zyx = np.asarray(vertices_zyx, dtype=np.float32)
        self.faces = np.asarray(faces, dtype=np.uint32)

        if normals_zyx is None:
            self.normals_zyx = np.zeros((0,3), dtype=np.int32)
        else:
            self.normals_zyx = np.asarray(normals_zyx, np.float32)

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
    def from_file(cls, path):
        """
        Alternate constructor.
        Read a mesh from .obj or .drc
        """
        ext = os.path.splitext(path)[1]
        if ext == '.drc':
            # Convert from draco to OBJ
            # Use a pipe to avoid the need for the hard disk
            draco_output_pipe = TemporaryNamedPipe('output.obj')
            cmd = f"draco_decoder -i {path} -o {draco_output_pipe.path}" 
            proc = subprocess.Popen(cmd, shell=True)
            try:
                with open(draco_output_pipe.path, 'rb') as obj_stream:
                    #with open('/tmp/wtf.obj', 'wb') as f:
                    #    f.write(obj_stream.read())
                    #assert False
                    vertices_zyx, faces, normals_zyx = read_obj(obj_stream)
                    proc.wait()
                return Mesh(vertices_zyx, faces, normals_zyx)
            finally:
                if proc.returncode != 0:
                    msg = f"Child process returned an error code: {proc.returncode}.\n"\
                          f"Command was: {cmd}"
                    logger.error(msg)
                    raise RuntimeError(msg)
        elif ext == '.obj':
            with open(path, 'rb') as obj_stream:
                vertices_zyx, faces, normals_zyx = read_obj(obj_stream)
                return Mesh(vertices_zyx, faces, normals_zyx)
        else:
            msg = f"Unknown file type: {path}"
            logger.error(msg)
            raise RuntimeError(msg)

        return Mesh

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
                msg = f"Uknown method: {method}"
                logger.error(msg)
                raise RuntimeError(msg)
        except ValueError:
            if downsampled_volume_zyx.all():
                # Completely full boxes are not meshable -- they would be
                # open on all sides, leaving no vertices or faces.
                # Just return an empty mesh.
                empty_vertices = np.zeros( (0, 3), dtype=np.float32 )
                empty_faces = np.zeros( (0, 3), dtype=np.uint32 )
                return Mesh(empty_vertices, empty_faces, box=fullres_box_zyx)
            else:
                logger.error("Error during mesh generation")
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


    def recompute_normals(self):
        self.normals_zyx = compute_vertex_normals(self.vertices_zyx, self.faces)
        

    def simplify(self, fraction, recompute_normals=True):
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
        self.normals_zyx = np.array((0,3), dtype=np.float32)
        
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

        if recompute_normals:
            self.recompute_normals()
        
        proc.wait(timeout=1.0)
        if proc.returncode != 0:
            msg = f"Child process returned an error code: {proc.returncode}.\n"\
                  f"Command was: {cmd}"
            logger.error(msg)
            raise RuntimeError(msg)


    def laplacian_smooth(self, iterations=1, recompute_normals=True):
        """
        Smooth the mesh in-place.
         
        This is simplest mesh smoothing technique, known as Laplacian Smoothing.
        Relocates each vertex by averaging its position with that of its adjacent neighbors.
        Repeat for N iterations.
        
        Disadvantage: Results in overall shrinkage of the mesh, especially for more iterations.

        Args:
            iterations:
                How many passes to take over the data.
                More iterations results in a smoother mesh, but more shrinkage (and more CPU time).
            
            recompute_normals:
                The previous normals are discarded.
                If recompute_normals=True, they will be recomputed after smoothing.
        
        TODO: Variations of this technique can give refined results.
            - Try weighting the influence of each neighbor by it's distance to the center vertex.
            - Try smaller displacement steps for each iteration
            - Try switching between 'push' and 'pull' iterations to avoid shrinkage
            - Try smoothing "boundary" meshes independently from the rest of the mesh (less shrinkage)
            - Try "Cotangent Laplacian Smoothing"
        """
        if iterations == 0:
            return
        
        # Always discard old normals
        self.normals_zyx = np.zeros((0,3), np.float32)

        # Compute the list of all unique vertex adjacencies
        all_edges = np.concatenate( [self.faces[:,(0,1)],
                                     self.faces[:,(1,2)],
                                     self.faces[:,(2,0)]] )
        all_edges.sort(axis=1)
        edges_df = pd.DataFrame( all_edges, columns=['v1_id', 'v2_id'] )
        edges_df.drop_duplicates(inplace=True)
        del all_edges

        # (This sort isn't technically necessary, but it might give
        # better cache locality for the vertex lookups below.)
        edges_df.sort_values(['v1_id', 'v2_id'], inplace=True)

        # How many neighbors for each vertex == how many times it is mentioned in the edge list
        neighbor_counts = np.bincount(edges_df.values.flat, minlength=len(self.vertices_zyx))
        
        new_vertices_zyx = np.empty_like(self.vertices_zyx)
        for _ in range(iterations):
            new_vertices_zyx[:] = self.vertices_zyx

            # For the complete edge index list, accumulate (sum) the vertexes on
            # the right side of the list into the left side's address and vice-versa.
            #
            ## We want something like this:
            # v1_indexes, v2_indexes = df['v1_id'], df['v2_id']
            # new_vertices_zyx[v1_indexes] += self.vertices_zyx[v2_indexes]
            # new_vertices_zyx[v2_indexes] += self.vertices_zyx[v1_indexes]
            #
            # ...but that doesn't work because v1_indexes will contain repeats,
            #    and "fancy indexing" behavior is undefined in that case.
            #
            # Instead, it turns out that np.ufunc.at() works (it's an "unbuffered" operation)
            np.add.at(new_vertices_zyx, edges_df['v1_id'], self.vertices_zyx[edges_df['v2_id'], :])
            np.add.at(new_vertices_zyx, edges_df['v2_id'], self.vertices_zyx[edges_df['v1_id'], :])

            new_vertices_zyx[:] /= (neighbor_counts[:,None] + 1) # plus one because each point itself is included in the sum

            # Swap (save RAM allocation overhead by reusing the new_vertices_zyx array between iterations)
            self.vertices_zyx, new_vertices_zyx = new_vertices_zyx, self.vertices_zyx
        
        if recompute_normals:
            self.recompute_normals()


    def serialize(self, path=None, fmt=None):
        """
        Serialize the mesh data in either .obj or .drc format.
        If path is given, write to that file.
        Otherwise, return the serialized data as a bytes object.
        """
        if path is not None:
            fmt = os.path.splitext(path)[1][1:]
        elif fmt is None:
            fmt = 'obj'
            
        assert fmt in ('obj', 'drc')
        
        if len(self.vertices_zyx) == 0:
            if path:
                open(path, 'wb').close()
                return
            return b''
        
        obj_bytes = write_obj(self.vertices_zyx, self.faces, self.normals_zyx)

        if fmt == 'obj':
            if path:
                with open(path, 'wb') as f:
                    f.write(obj_bytes)
            else:
                return obj_bytes

        elif fmt == 'drc':
            # Sadly, draco is incapable of reading from non-seekable inputs.
            # It requires an actual input file, so we can't use a named pipe to avoid disk I/O.
            # But at least we can use a pipe for the output...
            mesh_dir = AutoDeleteDir(tempfile.mkdtemp())
            mesh_path = f'{mesh_dir}/mesh.obj'
            with open(mesh_path, 'wb') as mesh_file, BytesIO(obj_bytes) as obj_stream:
                copyfileobj(obj_stream, mesh_file)

            global DEBUG_DRACO
            if DEBUG_DRACO:
                # Write the .drc file to disk and read it that way,
                # instead of using a PIPE to save RAM.
                drc_path = f"{mesh_dir}/mesh.drc"
                cmd = f'draco_encoder -cl 5 -i {mesh_path} -o {drc_path}'
                try:
                    subprocess.check_call(cmd, shell=True)
                except:
                    mesh_dir.skip_delete = True
                    raise
                with open(drc_path, 'rb') as f:
                    drc_bytes = f.read()
            else:
                # Use a unix 'named pipe' to receive the output from the
                # draco encoder without the need to write to the hard disk and read it.
                draco_output_pipe = TemporaryNamedPipe('output.drc')
                cmd = f'draco_encoder -cl 5 -i {mesh_path} -o {draco_output_pipe.path}'
                proc = subprocess.Popen(cmd, shell=True)
                with draco_output_pipe.open_stream('rb') as drc_stream:
                    drc_bytes = drc_stream.read()

                proc.wait(timeout=600.0) # 10 minutes
                if proc.returncode != 0:
                    msg = f"Child process returned an error code: {proc.returncode}.\n"\
                          f"Command was: {cmd}\n\n"
                    msg += "Input mesh was: \n\n"
                    msg += obj_bytes.decode()
                    logger.error(msg)
                    raise RuntimeError(msg)

            if path:
                with open(path, 'wb') as f:
                    f.write(drc_bytes)
            else:
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

