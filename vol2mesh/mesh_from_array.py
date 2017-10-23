import os
import tempfile
import threading
import subprocess
from io import BytesIO
from shutil import copyfileobj, rmtree

import numpy as np
from marching_cubes import march

class AutoDeleteDir:
    def __init__(self, dirpath):
        self.dirpath = dirpath
    
    def __del__(self):
        rmtree(self.dirpath)

    def __str__(self):
        return self.dirpath
    
class TemporaryNamedPipe:
    """
    Represents a unix 'named pipe', a.k.a. fifo
    The pipe is created in a temporary directory and deleted upon cleanup() (or __exit__).

    Example:

        with TemporaryNamedPipe() as pipe:
            def write_hello():
                with pipe.open_stream('r') as f:
                    f.write("Hello")
            threading.Thread(target=write_hello).start()
            
            subprocess.call(f'cat {pipe.path}')
    """
    def __init__(self, basename='temporary-pipe.bin'):
        self.state = 'uninitialized'
        assert '/' not in basename
        self.tmpdir = tempfile.mkdtemp()
        self.path = f"{self.tmpdir}/{basename}"
    
        os.mkfifo(self.path)
        self.state = 'pipe_exists'
        self.writer_thread = None
    
    def cleanup(self):
        if self.path:
            os.unlink(self.path)
            os.rmdir(self.tmpdir)
            self.path = None

    def __del__(self):
        self.cleanup()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.cleanup()

    def start_writing_stream(self, stream):
        def write_input():
            with open(self.path, 'wb') as f:
                copyfileobj(stream, f)
        self.writer_thread = threading.Thread(target=write_input)
        self.writer_thread.start()
        return self.writer_thread

    def open_stream(self, mode):
        return TemporaryNamedPipe.Stream(mode, self)
    
    class Stream:
        """
        An open stream to a parent TemporaryNamedPipe.
        Retains a reference to the parent so the pipe isn't deleted before stream is closed.
        """
        def __init__(self, mode, parent_pipe):
            self._parent_pipe = parent_pipe
            self._file = open(parent_pipe.path, mode)
            self.closed = False
            self.mode = mode
            self.name = self._file.name

        def close(self):
            if self._parent_pipe:
                self._file.close()
                self._parent_pipe = None # Release parent pipe
                self.closed = True

        def __del__(self):
            self.close()
            
        def __enter__(self):
            return self
        
        def __exit__(self, *args):
            self.close()

        def fileno(self, *args, **kwargs): return self._file.fileno(*args, **kwargs)
        def flush(self, *args, **kwargs): return self._file.flush(*args, **kwargs)
        def isatty(self, *args, **kwargs): return self._file.isatty(*args, **kwargs)
        def readable(self, *args, **kwargs): return self._file.readable(*args, **kwargs)
        def readline(self, *args, **kwargs): return self._file.readline(*args, **kwargs)
        def readlines(self, *args, **kwargs): return self._file.readlines(*args, **kwargs)
        def seek(self, *args, **kwargs): return self._file.seek(*args, **kwargs)
        def tell(self, *args, **kwargs): return self._file.tell(*args, **kwargs)
        def truncate(self, *args, **kwargs): return self._file.truncate(*args, **kwargs)
        def writable(self, *args, **kwargs): return self._file.writable(*args, **kwargs)

        def read(self, *args, **kwargs): return self._file.read(*args, **kwargs)
        def readall(self, *args, **kwargs): return self._file.readall(*args, **kwargs)
        def readinto(self, *args, **kwargs): return self._file.readinto(*args, **kwargs)
        def write(self, *args, **kwargs): return self._file.write(*args, **kwargs)


def generate_obj(vertices_xyz, faces):
    """
    Given lists of vertices and faces, write them to a BytesIO in .obj format.
    """
    mesh_bytes = BytesIO()
    mesh_bytes.write(b"# OBJ file\n")
    for (x,y,z) in vertices_xyz:
        mesh_bytes.write(f"v {x:.7f} {y:.7f} {z:.7f}\n".encode('utf-8'))
    for (v1, v2, v3) in faces:
        mesh_bytes.write(f"f {v1} {v2} {v3} \n".encode('utf-8'))
    
    mesh_bytes.seek(0)
    return mesh_bytes


def mesh_from_array(volume_zyx, box_zyx, downsample_factor=1, simplify_ratio=None, smoothing_rounds=3, output_format='obj'):
    """
    Given a binary volume, convert it to a mesh in .obj format, optionally simplified.
    
    volume_zyx: Binary volume (ZYX order)
    box: Bounding box of the the volume data in global non-downsampled coordinates [(z0,y0,x0), (z1,y1,x1)]
    downsample_factor: Factor by which the given volume has been downsampled from its original size
    simplify_ratio: How much to simplify the generated mesh (or None to skip simplification)
    smoothing_rounds: Passed to marching_cubes.march()
    output_format: Either 'drc' or 'obj'
    """
    assert output_format in ('obj', 'drc'), \
        f"Unknown output format: {output_format}.  Expected one of ('obj', 'drc')"

    if simplify_ratio == 1.0:
        simplify_ratio = None

    volume_xyz = volume_zyx.transpose()
    box_xyz = np.asarray(box_zyx)[:,::-1]

    vertices_xyz, _normals, faces = march(volume_xyz, smoothing_rounds)

    # Rescale and translate
    vertices_xyz[:] *= downsample_factor
    vertices_xyz[:] += box_xyz[0]
    
    # I don't understand why we write face vertices in reverse order...
    # ...does marching_cubes give clockwise order instead of counter-clockwise?
    # Is it because we passed a fortran-order array?
    faces = faces[:, ::-1]
    faces += 1

    mesh_stream = generate_obj(vertices_xyz, faces)
        
    child_processes = []

    try:
        if simplify_ratio is not None:
            simplify_input_pipe = TemporaryNamedPipe('input.obj')
            simplify_input_pipe.start_writing_stream(mesh_stream)
        
            simplify_output_pipe = TemporaryNamedPipe('output.obj')
        
            cmd = f'fq-mesh-simplify {simplify_input_pipe.path} {simplify_output_pipe.path} {simplify_ratio}'
            child_processes.append( (cmd, subprocess.Popen(cmd, shell=True) ) )
            mesh_stream = simplify_output_pipe.open_stream('rb')

        if output_format == 'drc':
            # Sadly, draco is incapable of reading from non-seekable inputs.
            # It requires an actual input file, so we can't use a named pipe to avoid disk I/O.
            # But at least we can use a pipe for the output...
            mesh_dir = AutoDeleteDir(tempfile.mkdtemp())
            mesh_path = f'{mesh_dir}/mesh.obj'
            with open(mesh_path, 'wb') as mesh_file:
                copyfileobj(mesh_stream, mesh_file)
            draco_output_pipe = TemporaryNamedPipe('output.drc')
    
            cmd = f'draco_encoder -cl 5 -i {mesh_path} -o {draco_output_pipe.path}'
            child_processes.append( (cmd, subprocess.Popen(cmd, shell=True) ) )
            mesh_stream = draco_output_pipe.open_stream('rb')

        return mesh_stream.read()

    finally:
        # Explicitly wait() for the child processes
        # (avoids a warning from subprocess.Popen.__del__)
        for cmd, proc in child_processes:
            proc.wait(timeout=1.0)
            if proc.returncode != 0:
                raise RuntimeError(f"Child process returned an error code: {proc.returncode}.\n"
                                   f"Command was: {cmd}")

