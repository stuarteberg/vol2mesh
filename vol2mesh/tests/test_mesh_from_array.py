from io import BytesIO
import unittest
import numpy as np
from scipy.ndimage import distance_transform_edt

from vol2mesh import mesh_from_array
from vol2mesh.obj_utils import read_obj

class Test_mesh_from_array(unittest.TestCase):
     
    def test(self):
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
        box = [(0,0,0), (binary_vol.shape)]

        # Does it run at all?
        mesh = mesh_from_array( binary_vol, box[0], 1, simplify_ratio=None )

        #with open('/tmp/test-mesh.obj', 'wb') as f:
        #    f.write(mesh)
  
        # Simplifying makes the file smaller
        mesh_simplified = mesh_from_array( binary_vol, box[0], 1, simplify_ratio=0.5 )
        assert len(mesh) > len(mesh_simplified), f"not true: {len(mesh)} > {len(mesh_simplified)}"
         
        # Simplifying more makes it even smaller
        mesh_more_simplified = mesh_from_array( binary_vol, box[0], 1, simplify_ratio=0.2 )
        assert len(mesh_simplified) > len(mesh_more_simplified), f"not true: {len(mesh_simplified)} > {len(mesh_more_simplified)}"
    
        # Running draco on top should be smaller than simplification
        mesh_compressed = mesh_from_array( binary_vol, box[0], 1, simplify_ratio=None, output_format='drc' )
        assert len(mesh_compressed) < len(mesh_more_simplified), "Draco-encoded mesh should be smaller"
         
        # Running both simplification and draco: even smaller 
        mesh_simple_compressed = mesh_from_array( binary_vol, box[0], 1, simplify_ratio=0.2, output_format='drc' )
        assert len(mesh_simple_compressed) < len(mesh_compressed), "Draco-encoded mesh should be smaller"

    def test_tiny_array(self):
        """
        Tiny arrays can't be meshified. An exception is raised.
        """
        one_voxel = np.ones((1,1,1), np.uint8)
        try:
            _tiny_mesh = mesh_from_array( one_voxel, (0,0,0), 1, simplify_ratio=None )
        except ValueError:
            pass
 
    def test_solid_array(self):
        """
        Solid volumes can't be meshified. An empty mesh is returned instead.
        """
        solid_volume = np.ones((3,3,3), np.uint8)
        mesh_bytes = mesh_from_array( solid_volume, (0,0,0), 1, simplify_ratio=None )
        assert len(mesh_bytes) == 0
 
    def test_simplify_single_padded_voxel(self):
        """
        The simplification tool rejects mesh/ratio combinations
        that would result in meshes that are very small.
        In those cases, the mesh is just left unsimplified.
        """
        one_padded_voxel = np.zeros((3,3,3), np.uint8)
        one_padded_voxel[1,1,1] = 1
        _tiny_mesh = mesh_from_array( one_padded_voxel, (0,0,0), 1, simplify_ratio=0.2, output_format='drc' )

    def test_normals_after_simplification(self):
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
        box = [(0,0,0), (binary_vol.shape)]
    
        # Simplified should still contain normals.
        # TODO: Check that they are, like, correct...
        mesh_simplified = mesh_from_array( binary_vol, box[0], 1, simplify_ratio=0.5 )
        vertexes, _faces, normals = read_obj(BytesIO(mesh_simplified))
        assert normals.shape == vertexes.shape

if __name__ == "__main__":
    unittest.main()
