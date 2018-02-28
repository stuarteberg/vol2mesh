from itertools import starmap
import unittest
import numpy as np
from scipy.ndimage import distance_transform_edt

from vol2mesh.mesh import Mesh, concatenate_meshes

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
#             f.write(mesh.serialize('drc'))

    def test_tiny_array(self):
        """
        Tiny arrays trigger an exception in skimage, so they must be padded first.
        Verify that they can be meshified (after padding).
        """
        one_voxel = np.ones((1,1,1), np.uint8)
        mesh = Mesh.from_binary_vol( one_voxel, [(0,0,0), (1,1,1)] )

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
        mesh.serialize()
        concatenate_meshes((mesh, mesh))
    
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
