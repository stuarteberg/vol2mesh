import unittest
import threading
import subprocess
from io import BytesIO
from vol2mesh.io_utils import TemporaryNamedPipe

import pytest

@pytest.mark.skip("Skipping broken io utils tests")
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

    def test_copy_stream(self):
        pipe = TemporaryNamedPipe()
        s = BytesIO(b'Hello')
        th = pipe.start_writing_stream(s)
        cat_output = subprocess.check_output(f'cat {pipe.path}', shell=True)
        assert cat_output == b"Hello"
        assert cat_output == b'Hello'
        th.join()

if __name__ == "__main__":
    unittest.main()
