import os
import glob
import logging
import subprocess
from io import BytesIO

import numpy as np
import pandas as pd
from skimage.measure import marching_cubes_lewiner

from dvidutils import remap_duplicates, LabelMapper, encode_faces_to_drc_bytes, decode_drc_bytes_to_faces

from .normals import compute_face_normals, compute_vertex_normals
from .obj_utils import write_obj, read_obj
from .io_utils import TemporaryNamedPipe

logger = logging.getLogger(__name__)

DRACO_USE_PIPE = False

class Mesh:
    """
    A class to hold the elements of a mesh.
    """
    def __init__(self, vertices_zyx, faces, normals_zyx=None, box=None, pickle_with_draco=True):
        """
        Args:
            vertices_zyx: ndarray (N,3), float
            
            faces: ndarray (M,3), integer
                Each element is an index referring to an element of vertices_zyx
        
            normals_zyx: ndarray (N,3), float
            
            box: ndarray (2,3)
                Overall bounding box of the mesh.
                (The bounding box information is not stored in mesh files like .obj and .drc,
                but it is useful to store it here for programmatic manipulation.)
            
            pickle_with_draco:
                If True, pickling will be performed by encoding the vertices, normals, and faces via draco compression.
        """
        self.pickle_with_draco = pickle_with_draco
        self._destroyed = False
        
        # Note: When restoring from pickled data, vertices and faces
        #       are restored lazily, upon first access.
        #       See __getstate__().
        self._vertices_zyx = np.asarray(vertices_zyx, dtype=np.float32)
        self._faces = np.asarray(faces, dtype=np.uint32)
        self._draco_bytes = None

        if normals_zyx is None:
            self._normals_zyx = np.zeros((0,3), dtype=np.int32)
        else:
            self._normals_zyx = np.asarray(normals_zyx, np.float32)
            assert self._normals_zyx.shape in (self.vertices_zyx.shape, (0,3)), \
                "Normals were provided, but they don't match the shape of the vertices:\n" \
                f" {self._normals_zyx.shape} != {self.vertices_zyx.shape}"

        for a in (self._vertices_zyx, self._faces, self._normals_zyx):
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
            with open(path, 'rb') as drc_stream:
                draco_bytes = drc_stream.read()
            return Mesh.from_buffer(draco_bytes, 'drc')

        elif ext == '.obj':
            with open(path, 'rb') as obj_stream:
                vertices_zyx, faces, normals_zyx = read_obj(obj_stream)
                return Mesh(vertices_zyx, faces, normals_zyx)
        else:
            msg = f"Unknown file type: {path}"
            logger.error(msg)
            raise RuntimeError(msg)


    @classmethod
    def from_directory(cls, path):
        """
        Alternate constructor.
        Read all mesh files (either .drc or .obj) from a
        directory and concatenate them into one big mesh.
        """
        mesh_paths = glob.glob(f'{path}/*.drc') + glob.glob(f'{path}/*.obj')
        meshes = map(Mesh.from_file, mesh_paths)
        return concatenate_meshes(meshes)


    @classmethod
    def from_buffer(cls, serialized_bytes, fmt):
        """
        Alternate constructor.
        Read a mesh from either .obj or .drc format, from a buffer.
        
        Args:
            serialized_bytes:
                bytes object containing the .obj or .drc file contents
            fmt:
                Either 'obj' or 'drc'.
        """
        assert fmt in ('obj', 'drc')
        if len(serialized_bytes) == 0:
            return Mesh(np.zeros((0,3), np.float32), np.zeros((0,3), np.uint32))

        if fmt == 'obj':
            with BytesIO(serialized_bytes) as obj_stream:
                vertices_zyx, faces, normals_zyx = read_obj(obj_stream)
                return Mesh(vertices_zyx, faces, normals_zyx)
        elif fmt == 'drc':
            vertices_xyz, normals_xyz, faces = decode_drc_bytes_to_faces(serialized_bytes)
            vertices_zyx = vertices_xyz[:,::-1]
            normals_zyx = normals_xyz[:,::-1]
            return Mesh(vertices_zyx, faces, normals_zyx)


    @classmethod
    def from_binary_vol(cls, downsampled_volume_zyx, fullres_box_zyx=None, method='skimage', step_size=1):
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

                vertices_zyx, faces, normals_zyx, _values = marching_cubes_lewiner(downsampled_volume_zyx, 0.5, step_size=step_size)
                
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
    def from_binary_blocks(cls, downsampled_binary_blocks, fullres_boxes_zyx, stitch=True, method='skimage'):
        """
        Alternate constructor.
        Compute a mesh for each of the given binary volumes
        (scaled and translated according to its associated box),
        and concatenate them (but not stitch them).
        
        Args:
            downsampled_binary_blocks:
                List of binary blocks on which to run marching cubes.
                The blocks need not be full-scale; their meshes will be re-scaled
                according to their corresponding bounding-boxes in fullres_boxes_zyx.

            fullres_boxes_zyx:
                List of bounding boxes corresponding to the blocks.
                Each block meshes will be re-scaled to fit exactly within it's bounding box.
            
            stitch:
                If True, deduplicate the vertices in the final mesh and topologically
                connect the faces in adjacent blocks.
            
            method:
                Which library to use for marching_cubes. Currently, only 'skimage' is supported.
        """
        meshes = []
        for binary_vol, fullres_box_zyx in zip(downsampled_binary_blocks, fullres_boxes_zyx):
            mesh = cls.from_binary_vol(binary_vol, fullres_box_zyx, method)
            meshes.append(mesh)

        mesh = concatenate_meshes(meshes)
        if stitch:
            mesh.stitch_adjacent_faces(drop_duplicate_vertices=True, drop_duplicate_faces=True)
        return mesh

    def compress(self):
        # Sadly, we MUST remove degenerate faces before compressing with draco,
        # whether or not normals are present, due to https://github.com/google/draco/issues/356
        # Maybe draco > 1.2.5 will fix the problem.
        # In the meantime, the easiest way to remove degenerate faces is to recompute the normals
        # every time we encode to draco.
        # (Technically, this workaround doesn't guard against ALMOST-degenerate faces, which draco might turn
        # into degenerate faces when it reduces the quantization of the vertices.  But it seems to avoid the segfault for now.) 
        if self.normals_zyx.shape[0] == 0:
            self.recompute_normals(True)
        
        if self._draco_bytes is None and len(self._vertices_zyx) > 0:
            self._draco_bytes = encode_faces_to_drc_bytes(self._vertices_zyx[:,::-1], self._normals_zyx[:,::-1], self._faces)
            self._vertices_zyx = None
            self._normals_zyx = None
            self._faces = None
        
        if self._draco_bytes is None:
            return 0
        return len(self._draco_bytes)
        
    def __getstate__(self):
        """
        Pickle representation.
        If self.pickle_with_draco is True, compress the mesh to a buffer with draco
        (vertices and faces only, for now), and discard the original arrays.
        """
        if self.pickle_with_draco:
            self.compress()
        return self.__dict__

    def destroy(self):
        """
        Clear the mesh data.
        Release all of our big members.
        Useful for spark workflows, in which you don't immediately 
        all references to the mesh, but you know you're done with it.
        """
        self._draco_bytes = None
        self._vertices_zyx = None
        self._faces = None
        self._normals_zyx = None
        self._destroyed = True

    @property
    def vertices_zyx(self):
        assert not self._destroyed
        if self._vertices_zyx is None:
            self._decode_from_pickled_draco()
        return self._vertices_zyx

    @vertices_zyx.setter
    def vertices_zyx(self, new_vertices_zyx):
        assert not self._destroyed
        self._vertices_zyx = new_vertices_zyx

    @property
    def faces(self):
        assert not self._destroyed
        if self._faces is None:
            self._decode_from_pickled_draco()
        return self._faces

    @faces.setter
    def faces(self, new_faces):
        assert not self._destroyed
        self._faces = new_faces

    @property
    def normals_zyx(self):
        assert not self._destroyed
        if self._normals_zyx is None:
            self._decode_from_pickled_draco()
        return self._normals_zyx
    
    @normals_zyx.setter
    def normals_zyx(self, new_normals_zyx):
        assert not self._destroyed
        self._normals_zyx = new_normals_zyx

    def _decode_from_pickled_draco(self):
        vertices_xyz, normals_xyz, self._faces = decode_drc_bytes_to_faces(self._draco_bytes)
        self.vertices_zyx = vertices_xyz[:, ::-1]
        self.normals_zyx = normals_xyz[:, ::-1]
        self._draco_bytes = None

    def stitch_adjacent_faces(self, drop_duplicate_vertices=True, drop_duplicate_faces=True):
        """
        Search for duplicate vertices and remove all references to them in self.faces,
        by replacing them with the index of the first matching vertex in the list.
        Works in-place.
        
        Note: Normals are discarded.  Call recompute_normals() if you need them.
        
        Args:
            drop_duplicate_vertices:
                If True, drop the duplicate vertices from self.vertices_zyx
                (since no faces refer to them any more, this saves some RAM).
            
            drop_duplicate_faces:
                If True, remove faces with an identical
                vertex list to any previous face.
            
            recompute_normals:
                Normals are discarded during the stitching procedure.
                To recompute them afterwards, set recompute_normals=True.
        """
        mapping_pairs = remap_duplicates(self.vertices_zyx)
        
        dup_indices, orig_indices = mapping_pairs.transpose()
        if len(dup_indices) == 0:
            return

        # Discard old normals
        self.normals_zyx = np.zeros((0,3), np.float32)

        # Remap faces to no longer refer to the duplicates
        mapper = LabelMapper(dup_indices, orig_indices)
        mapper.apply_inplace(self.faces, allow_unmapped=True)
        del mapper
        del orig_indices
        
        # Now the faces have been stitched, but the duplicate
        # vertices are still unnecessarily present,
        # and the face vertex indexes still reflect that.
        # Also, we may have uncovered duplicate faces now that the
        # vertexes have been canonicalized.

        def _drop_duplicate_vertices():
            # Calculate shift:
            # Determine number of duplicates above each vertex in the list
            drop_mask = np.zeros((self.vertices_zyx.shape[0]), bool)
            drop_mask[(dup_indices, )] = True
            cumulative_dupes = np.zeros(drop_mask.shape[0]+1, np.uint32)
            np.add.accumulate(drop_mask, out=cumulative_dupes[1:])
    
            # Renumber the faces
            orig = np.arange(len(self.vertices_zyx), dtype=np.uint32)
            shiftmap = orig - cumulative_dupes[:-1]
            self.faces = shiftmap[self.faces]

            # Delete the duplicate vertexes, which are unused now
            self.vertices_zyx = np.delete(self.vertices_zyx, dup_indices, axis=0)

        if drop_duplicate_vertices:
            _drop_duplicate_vertices()

        def _drop_duplicate_faces():
            # Normalize face vertex order before checking for duplicates.
            # Technically, this means we don't distinguish
            # betweeen clockwise/counter-clockwise ordering,
            # but that seems unlikely to be a problem in practice.
            sorted_faces = pd.DataFrame(np.sort(self.faces, axis=1))
            duplicate_faces_mask = sorted_faces.duplicated()
            faces_df = pd.DataFrame(self.faces)
            faces_df.drop(duplicate_faces_mask.nonzero()[0], inplace=True)
            self.faces = np.asarray(faces_df.values, order='C')

        if drop_duplicate_faces:
            _drop_duplicate_faces()


    def recompute_normals(self, remove_degenerate_faces=True):
        """
        Compute the normals for this mesh.
        
        remove_degenerate_faces:
            If True, faces with no area (i.e. just lines) will be removed.
            (They have no effect on the vertex normals either way.)
        """
        face_normals = compute_face_normals(self.vertices_zyx, self.faces)

        if remove_degenerate_faces:
            # Degenerate faces ended up with a normal of 0,0,0.  Remove those faces.
            # (Technically, we might be left with unused vertices after this,
            #  but removing them requires relabeling the faces.
            #  Call stitch_adjacent_faces() if you want to remove them.)
            good_faces = face_normals.any(axis=1)
            if not good_faces.all():
                self.faces = self.faces[good_faces, :]
            del good_faces

        if len(self.faces) == 0:
            # No faces left. Discard all remaining vertices and normals.
            self.vertices_zyx = np.zeros((0,3), np.float32)
            self.normals_zyx = np.zeros((0,3), np.float32)
        else:
            self.normals_zyx = compute_vertex_normals(self.vertices_zyx, self.faces, face_normals=face_normals)
        

    def simplify(self, fraction):
        """
        Simplify this mesh in-place, by the given fraction (of the original vertex count).
        
        Note: Normals are discarded.  Call recompute_normals() if you need them.
        """
        # Normals are about to get discarded and recomputed anyway,
        # so delete them now to save some RAM and serialization time.
        self.normals_zyx = np.zeros((0,3), dtype=np.float32)
        
        # The fq-mesh-simplify tool rejects inputs that are too small (if the decimated face count would be less than 4).
        # We have to check for this in advance because we can't gracefully handle the error.
        # https://github.com/neurolabusc/Fast-Quadric-Mesh-Simplification-Pascal-/blob/master/c_code/Main.cpp
        if fraction is None or fraction == 1.0 or (len(self.faces) * fraction <= 4):
            return self

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

        proc.wait(timeout=1.0)
        if proc.returncode != 0:
            msg = f"Child process returned an error code: {proc.returncode}.\n"\
                  f"Command was: {cmd}"
            logger.error(msg)
            raise RuntimeError(msg)


    def laplacian_smooth(self, iterations=1):
        """
        Smooth the mesh in-place.
         
        This is simplest mesh smoothing technique, known as Laplacian Smoothing.
        Relocates each vertex by averaging its position with those of its adjacent neighbors.
        Repeat for N iterations.
        
        Disadvantage: Results in overall shrinkage of the mesh, especially for many iterations.
                      (But nearly all smoothing techniques cause at least some shrinkage.)

        Note: Normals are discarded.  Call recompute_normals() if you need them.
        
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
            
        assert fmt in ('obj', 'drc'), f"Unknown format: {fmt}"

        # Shortcut for empty mesh
        # Returns an empty buffer regardless of output format        
        empty_mesh = (self._draco_bytes is not None and self._draco_bytes == b'') or len(self.vertices_zyx)
        if empty_mesh == 0:
            if path:
                open(path, 'wb').close()
                return
            return b''

        if fmt == 'obj':
            if path:
                with open(path, 'wb') as f:
                    write_obj(self.vertices_zyx, self.faces, self.normals_zyx, f)
            else:
                return write_obj(self.vertices_zyx, self.faces, self.normals_zyx)

        elif fmt == 'drc':
            draco_bytes = self._draco_bytes
            if draco_bytes is None:
                if self.normals_zyx.shape[0] == 0:
                    self.recompute_normals(True) # See comment in Mesh.compress()
                draco_bytes = encode_faces_to_drc_bytes(self.vertices_zyx[:,::-1], self.normals_zyx[:,::-1], self.faces)
            
            if path:
                with open(path, 'wb') as f:
                    f.write(draco_bytes)
            else:
                return draco_bytes


def concatenate_meshes(meshes):
    """
    Combine the given list of Mesh objects into a single Mesh object,
    renumbering the face vertices as needed, and expanding the bounding box
    to encompass the union of the meshes.
    """
    if not isinstance(meshes, list):
        meshes = list(meshes)

    vertex_counts = np.fromiter((len(mesh.vertices_zyx) for mesh in meshes), np.int64, len(meshes))
    face_counts = np.fromiter((len(mesh.faces) for mesh in meshes), np.int64, len(meshes))

    _verify_concatenate_inputs(meshes, vertex_counts)

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

def _verify_concatenate_inputs(meshes, vertex_counts):
    normals_counts = np.fromiter((len(mesh.normals_zyx) for mesh in meshes), np.int64, len(meshes))
    if not normals_counts.any() or (vertex_counts == normals_counts).all():
        # Looks good
        return

    # Uh-oh, we have a problem:
    # Either some meshes have normals while others don't, or some meshes
    # have normals that don't even match their OWN vertex count!
        
    import socket
    hostname = socket.gethostname()

    mismatches = (vertex_counts != normals_counts).nonzero()[0]

    msg = ("Mesh normals do not correspond to vertices.\n"
           "(Either exclude all normals, more make sure they match the vertices in every mesh.)\n"
           f"There were {len(mismatches)} mismatches out of {len(meshes)}\n")

    bad_mismatches = (normals_counts != vertex_counts) & (normals_counts != 0)
    if bad_mismatches.any():
        # Mismatches where the normals and vertices didn't even line up in the same mesh.
        # This should never happen.
        first_bad_mismatch = bad_mismatches.nonzero()[0][0]
        mesh = meshes[first_bad_mismatch]
        output_path = f'/tmp/BAD-mismatched-mesh-v{mesh.vertices_zyx.shape[0]}-n{mesh.normals_zyx.shape[0]}-{first_bad_mismatch}.obj'
        mesh.serialize(output_path)
        msg += f"Wrote first BAD mismatched mesh to {output_path} (host: {hostname})\n"
    
    missing_normals = (normals_counts != vertex_counts) & (normals_counts == 0)
    if missing_normals.any():
        # Mismatches where the normals and vertices didn't even line up in the same mesh.
        # This should never happen.
        first_missing_normals = missing_normals.nonzero()[0][0]
        output_path = f'/tmp/mismatched-mesh-no-normals-{first_missing_normals}.obj'
        meshes[first_missing_normals].serialize(output_path)
        msg += f"Wrote first mismatched (missing normals) mesh to {output_path} (host: {hostname})\n"
    
    matching_meshes = (normals_counts == vertex_counts) & (normals_counts > 0)
    if matching_meshes.any():
        first_matching_mesh = matching_meshes.nonzero()[0][0]
        output_path = f'/tmp/first-matching-mesh-{first_matching_mesh}.obj'
        meshes[first_matching_mesh].serialize(output_path)
        msg += f"Wrote first matching mesh to {output_path} (host: {hostname})\n"
    
    raise RuntimeError(msg)
