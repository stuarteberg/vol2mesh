import unittest
import numpy as np
from scipy.ndimage import distance_transform_edt

from vol2mesh.mesh import Mesh, concatenate_meshes

class TestMesh(unittest.TestCase):

    def setUp(self):
        # Create a test object (shaped like an 'X')
        center_line_img = np.zeros((100,100,100), dtype=np.uint32)
        for i in range(100):
            center_line_img[i, i, i] = 1
            center_line_img[99-i, i, i] = 1
         
        # Scipy distance_transform_edt conventions are opposite of vigra:
        # it calculates distances of non-zero pixels to the zero pixels.
        center_line_img = 1 - center_line_img
        distance_to_line = distance_transform_edt(center_line_img)
        binary_vol = (distance_to_line <= 10).astype(np.uint8)

        #binary_vol = np.pad(binary_vol, 1, 'constant', constant_values=0)
        data_box = [(0,0,0), (binary_vol.shape)]
        
        self.binary_vol = binary_vol
        self.data_box = data_box

    def test(self):
        mesh = Mesh.from_binary_vol( self.binary_vol, self.data_box )
        
        data_box = np.array(self.data_box)
        mesh_box = np.array([mesh.vertices_zyx.min(axis=0), 1+mesh.vertices_zyx.max(axis=0)])
        assert (mesh_box == data_box).all(), f"{mesh_box.tolist()} != {data_box.tolist()}"

    def test_blockwise(self):
        blocks = []
        boxes = []
        for z in range(0,100,20):
            for y in range(0,100,20):
                for x in range(0,100,20):
                    block = self.binary_vol[z:z+20, y:y+20, x:x+20]
                    if block.any():
                        blocks.append(block)
                        boxes.append( [(z,y,x), (z+20, y+20, x+20)] )
        
        mesh = Mesh.from_binary_blocks(blocks, boxes)
        data_box = np.array(self.data_box)
        mesh_box = np.array([mesh.vertices_zyx.min(axis=0), 1+mesh.vertices_zyx.max(axis=0)])
        assert (mesh_box == data_box).all(), f"{mesh_box.tolist()} != {data_box.tolist()}"
        

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
