import unittest
import numpy as np
from scipy.ndimage import distance_transform_edt

import threading
import subprocess
from vol2mesh.vol2mesh import mesh_from_array, TemporaryNamedPipe

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
        box = [(0,0,0), (binary_vol.shape)]
 
        # Does it run at all?
        mesh = mesh_from_array( binary_vol, box, 1, simplify_ratio=None, smoothing_rounds=0 )
 
        # Simplifying makes the file smaller
        mesh_simplified = mesh_from_array( binary_vol, box, 1, simplify_ratio=0.5, smoothing_rounds=0 )
        assert len(mesh) > len(mesh_simplified), f"not true: {len(mesh)} > {len(mesh_simplified)}"
 
        # Simplifying more makes it even smaller
        mesh_more_simplified = mesh_from_array( binary_vol, box, 1, simplify_ratio=0.2, smoothing_rounds=0 )
        assert len(mesh_simplified) > len(mesh_more_simplified), f"not true: {len(mesh_simplified)} > {len(mesh_more_simplified)}"
 
        # Smoothing (no simplification) makes the file... smaller?
        mesh_smoothed = mesh_from_array( binary_vol, box, 1, simplify_ratio=None, smoothing_rounds=10 )
        assert len(mesh) > len(mesh_smoothed), f"not true: {len(mesh)} > {len(mesh_smoothed)}"
 
class TestTemporaryNamedPipe(unittest.TestCase):
    def test_example(self):
        with TemporaryNamedPipe() as pipe:
            def write_hello():
                with pipe.open_stream('w') as f:
                    f.write("Hello")
            threading.Thread(target=write_hello).start()
              
            cat_output = subprocess.check_output(f'cat {pipe.path}', shell=True)
            assert cat_output == b"Hello"
 
    def test_read_from_unnamed_stream(self):
        pipe = TemporaryNamedPipe()
        def write_hello():
            with pipe.open_stream('w') as f:
                f.write("Hello")
        threading.Thread(target=write_hello).start()
  
        # The first cut of TemporaryNamedPipe.Stream() failed this test.
        text = pipe.open_stream('r').read()
        assert text == "Hello"


if __name__ == "__main__":
    unittest.main()
