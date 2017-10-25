import unittest
import threading
import subprocess
from vol2mesh.io_utils import TemporaryNamedPipe
 
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
