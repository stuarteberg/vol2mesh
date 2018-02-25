import unittest
import numpy as np
from vol2mesh.mesh_utils import Mesh, concatenate_meshes

class TestMeshUtil(unittest.TestCase):
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
