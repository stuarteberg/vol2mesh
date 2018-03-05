from itertools import starmap
import unittest
import numpy as np
from scipy.ndimage import distance_transform_edt

from vol2mesh.mesh import Mesh, concatenate_meshes

import vol2mesh.mesh
vol2mesh.mesh.DEBUG_DRACO = True

def box_to_slicing(start, stop):
    """
    For the given bounding box (start, stop),
    return the corresponding slicing tuple.

    Example:
    
        >>> assert bb_to_slicing([1,2,3], [4,5,6]) == np.s_[1:4, 2:5, 3:6]
    """
    return tuple( starmap( slice, zip(start, stop) ) )

class TestMesh(unittest.TestCase):

    def setUp(self):
        # Create a test object (shaped like an 'X')
        center_line_img = np.zeros((98,98,98), dtype=np.uint32)
        for i in range(98):
            center_line_img[i, i, i] = 1
            center_line_img[97-i, i, i] = 1

        # Scipy distance_transform_edt conventions are opposite of vigra:
        # it calculates distances of non-zero pixels to the zero pixels.
        center_line_img = 1 - center_line_img
        distance_to_line = distance_transform_edt(center_line_img)
        binary_vol = (distance_to_line <= 10).astype(np.uint8)

        binary_vol = np.pad(binary_vol, 1, 'constant')
        assert binary_vol.shape == (100,100,100)

        #binary_vol = np.pad(binary_vol, 1, 'constant', constant_values=0)
        data_box = [(0,0,0), (binary_vol.shape)]
        
        self.binary_vol = binary_vol
        self.data_box = data_box
        
        min_nonzero_coord = np.transpose(binary_vol.nonzero()).min(axis=0)
        max_nonzero_coord = np.transpose(binary_vol.nonzero()).max(axis=0)
        
        self.nonzero_box = np.array( [min_nonzero_coord, 1+max_nonzero_coord] )

    def test(self):
        # Pretend the data was downsampled and translated,
        # and therefore the mesh requires upscaling and translation
        data_box = np.array(self.data_box)
        data_box += 1000
        
        nonzero_box = self.nonzero_box + 1000
        
        FACTOR = 2
        data_box *= FACTOR
        nonzero_box *= FACTOR
        
        mesh = Mesh.from_binary_vol( self.binary_vol, data_box )
        assert mesh.vertices_zyx.dtype == np.float32
        
        mesh_box = np.array([mesh.vertices_zyx.min(axis=0),
                             mesh.vertices_zyx.max(axis=0)])
        assert (mesh_box == nonzero_box).all(), f"{mesh_box.tolist()} != {nonzero_box.tolist()}"
        
        serialized = mesh.serialize(fmt='obj')
        unserialized = mesh.from_buffer(serialized, 'obj')
        assert len(unserialized.vertices_zyx) == len(mesh.vertices_zyx)
        
        serialized = mesh.serialize(fmt='drc')
        unserialized = mesh.from_buffer(serialized, 'drc')
        assert len(unserialized.vertices_zyx) == len(mesh.vertices_zyx)
        

    def test_blockwise(self):
        data_box = np.array(self.data_box)
        blocks = []
        boxes = []
        for z in range(0,100,20):
            for y in range(0,100,20):
                for x in range(0,100,20):
                    OVERLAP = 1
                    box = np.asarray([(z,y,x), (z+20, y+20, x+20)], dtype=int)
                    box[0] -= OVERLAP
                    box[1] += OVERLAP
                    box = np.maximum(box, 0)
                    box = np.minimum(box, 1+data_box[1])

                    block = self.binary_vol[box_to_slicing(*box)]
                    if block.any():
                        blocks.append(block)
                        boxes.append( box )
        
        mesh = Mesh.from_binary_blocks(blocks, boxes)
        data_box = np.array(self.data_box)
        mesh_box = np.array([mesh.vertices_zyx.min(axis=0), mesh.vertices_zyx.max(axis=0)])
        assert (mesh_box == self.nonzero_box).all(), f"{mesh_box.tolist()} != {self.nonzero_box.tolist()}"
    
#         with open('/tmp/test-mesh.obj', 'wb') as f:
#             f.write(mesh.serialize())
# 
#         mesh.simplify(0.01)
#         with open('/tmp/test-mesh-simplified.obj', 'wb') as f:
#             f.write(mesh.serialize())
#         
#         with open('/tmp/test-mesh-simplified.drc', 'wb') as f:
#             f.write(mesh.serialize(fmt='drc'))

    def test_tiny_array(self):
        """
        Tiny arrays trigger an exception in skimage, so they must be padded first.
        Verify that they can be meshified (after padding).
        """
        one_voxel = np.ones((1,1,1), np.uint8)
        _mesh = Mesh.from_binary_vol( one_voxel, [(0,0,0), (1,1,1)] )

    def test_solid_array(self):
        """
        Solid volumes can't be meshified. An empty mesh is returned instead.
        """
        box = [(0,0,0), (3,3,3)]
        solid_volume = np.ones((3,3,3), np.uint8)
        
        mesh = Mesh.from_binary_vol( solid_volume, box )
        assert mesh.vertices_zyx.shape == (0,3)
        assert mesh.faces.shape == (0,3)
        assert mesh.normals_zyx.shape == (0,3)
        assert (mesh.box == box).all()

    def test_empty_mesh(self):
        """
        What happens when we call functions on an empty mesh?
        """
        mesh = Mesh( np.zeros((0,3), np.float32), np.zeros((0,3), int) )
        mesh.simplify(0.1)
        mesh.laplacian_smooth(2)
        mesh.serialize(fmt='obj')
        mesh.serialize(fmt='drc')
        concatenate_meshes((mesh, mesh))

    def test_smoothing_trivial(self):
        vertices_zyx = np.array([[0.0, 0.0, 0.0],
                                 [0.0, 0.0, 1.0],
                                 [0.0, 0.0, 2.0]])
 
        # This "face" is actually straight line,
        # which makes it easy to see what's going on
        faces = np.array([[0,1,2]])
        mesh = Mesh(vertices_zyx, faces)
        average_vertex = vertices_zyx.sum(axis=0) / 3
        mesh.laplacian_smooth(1)
        assert (mesh.vertices_zyx == average_vertex).all()

    def test_smoothing_hexagon(self):
        """
        Try 'smoothing' a simple 2D hexagon, which is an easy case to understand.
        """
        # This map is correctly labeled with the vertex indices
        _ = -1
        hexagon = [[[_,_,_,_,_,_,_],
                    [_,_,0,_,1,_,_],
                    [_,_,_,_,_,_,_],
                    [_,2,_,3,_,4,_],
                    [_,_,_,_,_,_,_],
                    [_,_,5,_,6,_,_],
                    [_,_,_,_,_,_,_]]]

        hexagon = 1 + np.array(hexagon)
        original_vertices_zyx = np.transpose(hexagon.nonzero())
        faces = [[3,1,4],
                 [3,4,6],
                 [3,6,5],
                 [3,5,2],
                 [3,2,0],
                 [3,0,1]]
        
        mesh = Mesh(original_vertices_zyx, faces)
        #mesh.serialize('/tmp/hexagon.obj')

        mesh.laplacian_smooth(1)
        #mesh.serialize('/tmp/hexagon-smoothed.obj')
        
        # Since vertex 3 is exactly centered between the rest,
        # it's location never changes.
        assert  (mesh.vertices_zyx[3] == original_vertices_zyx[3]).all()


    def test_smoothing_X(self):
        """
        This just exercises the code on our standard X-shaped
        test object, but doesn't verify the results.
        
        Uncomment the serialize() lines to inspect the effects manually.
        """
        mesh = Mesh.from_binary_vol( self.binary_vol, self.data_box )
        #mesh.serialize('/tmp/x-unsmoothed.obj')

        mesh.simplify(0.2)
        mesh.laplacian_smooth(5)        
        #mesh.serialize('/tmp/x-simplified-smoothed.obj')

        mesh = Mesh.from_binary_vol( self.binary_vol, self.data_box )
        mesh.laplacian_smooth(5)
        mesh.simplify(0.2)
        #mesh.serialize('/tmp/x-smoothed-simplified.obj')

    def test_stitch(self):
        vertices = np.zeros( (10,3), np.float32 )
        vertices[:,0] = np.arange(10)
        
        # Make 3 and 6 duplicates of 2 and 4, respectively
        vertices[3] = vertices[2]
        vertices[6] = vertices[4]
        
        faces = [[0,1,2],
                 [3,4,5],
                 [6,7,8]]
        
        remapped_faces = [[0,1,2],
                          [2,4,5], # After mapping away from dupes
                          [4,7,8]]
        
        remapped_faces = np.array(remapped_faces)
        
        # After dropping dupe rows
        remapped_faces[(remapped_faces > 6)] -= 1
        remapped_faces[(remapped_faces > 3)] -= 1
        
        reduced_vertices = list(vertices)
        del reduced_vertices[6]
        del reduced_vertices[3]
        
        mesh = Mesh(vertices, faces)
        mesh.stitch_aligned_faces()
        
        assert (mesh.faces == remapped_faces).all()
        assert (mesh.vertices_zyx == reduced_vertices).all()

class TestConcatenate(unittest.TestCase):

    def test_concatenate(self):
        vertexes_1 = np.array([[0,0,0],
                               [0,1,0],
                               [0,1,1]])
        faces_1 = np.array([[2,1,0]])

        vertexes_2 = np.array([[0,0,1],
                               [0,2,0],
                               [0,2,2]])

        faces_2 = np.array([[2,1,0],
                            [1,2,0]])

        vertexes_3 = np.array([[1,0,1],
                               [1,2,0],
                               [1,2,2]])

        faces_3 = np.array([[1,2,0]])

        mesh_1 = Mesh(vertexes_1, faces_1)
        mesh_2 = Mesh(vertexes_2, faces_2)
        mesh_3 = Mesh(vertexes_3, faces_3)
        
        combined_mesh = concatenate_meshes((mesh_1, mesh_2, mesh_3))
        assert (combined_mesh.vertices_zyx == np.concatenate((vertexes_1, vertexes_2, vertexes_3))).all()
        
        expected_faces = np.concatenate((faces_1, faces_2, faces_3))
        expected_faces[len(faces_1):] += len(vertexes_1)
        expected_faces[len(faces_1)+len(faces_2):] += len(vertexes_2)

        assert (combined_mesh.faces == expected_faces).all()

if __name__ == "__main__":
    unittest.main()
