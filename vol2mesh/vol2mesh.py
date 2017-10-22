import os
import sys
import glob
import tempfile
import threading
import subprocess
from io import BytesIO, RawIOBase
from shutil import copyfileobj
from contextlib import contextmanager

import numpy as np
import tifffile
from marching_cubes import march


SCALEX = 1.0
SCALEY = 1.0
SCALEZ = 1.0

def findBBDimensions(listOfPixels):
    xs = listOfPixels[0]
    ys = listOfPixels[1]
    zs = listOfPixels[2]

    minxs = min(xs)
    maxxs = max(xs)

    minys = min(ys)
    maxys = max(ys)

    minzs = min(zs)
    maxzs = max(zs)

    dx = maxxs - minxs
    dy = maxys - minys
    dz = maxzs - minzs

    return [minxs, maxxs+1, minys, maxys+1, minzs, maxzs+1], [dx, dy, dz]

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


def mesh_from_array(volume_zyx, box_zyx, downsample_factor=1, simplify_ratio=None, smoothing_rounds=3):
    """
    Given a binary volume, convert it to a mesh in .obj format, optionally simplified.
    
    volume_zyx: Binary volume (ZYX order)
    box: Bounding box of the the volume data in global non-downsampled coordinates [(z0,y0,x0), (z1,y1,x1)]
    downsample_factor: Factor by which the given volume has been downsampled from its original size
    simplify_ratio: How much to simplify the generated mesh (or None to skip simplification)
    smoothing_rounds: Passed to marching_cubes.march()
    """
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

    orig_mesh_stream = generate_obj(vertices_xyz, faces)

    if not simplify_ratio:
        return orig_mesh_stream.read()

    orig_mesh_pipe = TemporaryNamedPipe('original_mesh.obj')
    simplified_pipe = TemporaryNamedPipe('simlified_mesh.obj')
    orig_mesh_pipe.start_writing_stream(orig_mesh_stream)

    cmd_format = f'fq-mesh-simplify {orig_mesh_pipe.path} {simplified_pipe.path} {simplify_ratio}'
    simplify_proc = subprocess.Popen(cmd_format, shell=True)
    simplified_bytes = simplified_pipe.open_stream('rb').read()
    simplify_proc.wait()
    return simplified_bytes
    

def calcMeshWithCrop(stackname, labelStack, location, simplify, tags):
    print(str(tags['downsample_interval_x']))
    SCALEX = tags['downsample_interval_x']
    SCALEY = tags['downsample_interval_x']
    SCALEZ = tags['downsample_interval_x']
    indices = np.where(labelStack>0)
    box, dimensions = findBBDimensions(indices)


    window = labelStack[box[0]:box[1], box[2]:box[3], box[4]:box[5]]
    localIndices = np.where(window > 0)

    paddedWindowSizeList = list(window.shape)
    paddedWindowSize = tuple([i+2 for i in paddedWindowSizeList])

    blankImg = np.zeros(paddedWindowSize, dtype=bool)

    blankImg[tuple(([i+1 for i in localIndices[0]], [i+1 for i in localIndices[1]], [i+1 for i in localIndices[2]]))] = 1
    print("Building mesh...")
    vertices, normals, faces = march(blankImg.transpose(), 3)  # zero smoothing rounds
    with open(location + os.path.basename(stackname) +".obj", 'w') as f:
        f.write("# OBJ file\n")

        for v in vertices:
            f.write("v %.2f %.2f %.2f \n" % ((box[0] * SCALEX) + ((float(tags['dvid_offset_x']) + v[0]) * SCALEX), (box[2] * SCALEY) + ((float(tags['dvid_offset_x']) + v[1]) * SCALEY), (box[4] * SCALEZ) + (float(tags['dvid_offset_x']) + v[2]) * SCALEZ))
        #for n in normals:
            #f.write("vn -1 -1 -1 \n")# % (n[2], n[1], n[0]))
        for face in faces:
            f.write("f %d %d %d \n" % (face[2]+1, face[1]+1, face[0]+1))
    print("Decimating Mesh...")
    
    input_path = "./" + location + os.path.basename(stackname) +".obj"
    output_path = "./" + location + os.path.basename(stackname) +".smooth.obj"
    cmd = f'fq-mesh-simplify "{input_path}" "{output_path}" {simplify}'
    
    print(cmd)
    subprocess.call(cmd, shell=True)

def calcMesh(stackname, labelStack, location, simplify_ratio):
    tags = getTagDictionary(stackname)
    downsampleFactor = float(tags['downsample_interval_x'])
    xOffset = float(tags['dvid_offset_x'])
    yOffset = float(tags['dvid_offset_y'])
    zOffset = float(tags['dvid_offset_z'])
    labelStack = np.swapaxes(labelStack, 0, 2)
    print("Building mesh...")
    vertices, normals, faces = march(labelStack, 3)  # 3 smoothing rounds
    
    print('preparing vertices and faces...')
    vertStrings = ["v %.3f %.3f %.3f \n" % ((xOffset + i[0]) * downsampleFactor, (yOffset + i[1]) * downsampleFactor, (zOffset + i[2]) * downsampleFactor) for i in vertices]
    faceStrings = ["f %d %d %d \n" % (face[2]+1, face[1]+1, face[0]+1) for face in faces]
    with open(location + os.path.basename(stackname) +".obj", 'w') as f:
        f.write("# OBJ file\n")
        print("writing vertices...")
        f.write(''.join(vertStrings))
        #for n in normals:
        #    f.write("vn %.2f %.2f %.2f \n" % (n[2], n[1], n[0]))
        print("writing faces...")
        f.write(''.join(faceStrings))
    print("Decimating Mesh...")

    input_path = "./" + location + os.path.basename(stackname) +".obj"
    output_path = "./" + location + os.path.basename(stackname) +".smooth.obj"
    cmd = f'fq-mesh-simplify "{input_path}" "{output_path}" {simplify_ratio}'
    
    print(cmd)
    subprocess.call(cmd, shell=True)


def calcMeshWithOffsets(stackname, labelStack, location, simplify):
    tags = getTagDictionary(stackname)
    downsampleFactor = float(tags['downsample_interval_x'])
    xOffset = float(tags['dvid_offset_x'])
    yOffset = float(tags['dvid_offset_y'])
    zOffset = float(tags['dvid_offset_z'])
    
    labelStack = np.swapaxes(labelStack, 0, 2)
    print("Building mesh...")
    vertices, normals, faces = march(labelStack, 3)  # 3 smoothing rounds
    
    print('preparing vertices and faces...')
    vertStrings = ["v %.3f %.3f %.3f \n" % ((xOffset + i[0]) * downsampleFactor, (yOffset + i[1]) * downsampleFactor, (zOffset + i[2]) * downsampleFactor) for i in vertices]
    faceStrings = ["f %d %d %d \n" % (face[2]+1, face[1]+1, face[0]+1) for face in faces]
    with open(location + os.path.basename(stackname) +".obj", 'w') as f:
        f.write("# OBJ file\n")
        print("writing vertices...")
        f.write(''.join(vertStrings))
        #for n in normals:
        #    f.write("vn %.2f %.2f %.2f \n" % (n[2], n[1], n[0]))
        print("writing faces...")
        f.write(''.join(faceStrings))
    print("Decimating Mesh...")

    s = 'fq-mesh-simplify' + ' ./' + location + os.path.basename(stackname) +".obj ./" + location + os.path.basename(stackname) +".smooth.obj " + str(simplify)
    print(s)
    subprocess.call(s, shell=True)

def getTagDictionary(stack):
    tagDict = {}
    tif = tifffile.TiffFile(stack)
    tags = tif.pages[0].tags
    tagSet = []
    for page in tif.pages:
        try:
            tagDict['dvid_offset_x'] = page.tags['31232'].value

        except KeyError as e:
            pass
        try:
            tagDict['dvid_offset_y'] = page.tags['31233'].value
        except KeyError as e:
            pass
        try:
            tagDict['dvid_offset_z'] = page.tags['31234'].value
        except KeyError as e:
            pass
        try:
            tagDict['downsample_interval_x'] = float(page.tags['31235'].value) + 1.0
        except KeyError as e:
            pass
    if 'downsample_interval_x' not in tagDict:
        tagDict['downsample_interval_x'] = 1.0
    if 'dvid_offset_x' not in tagDict:
        tagDict['dvid_offset_x'] = 0.0
    if 'dvid_offset_y' not in tagDict:
        tagDict['dvid_offset_y'] = 0.0
    if 'dvid_offset_z' not in tagDict:
        tagDict['dvid_offset_z'] = 0.0

    return tagDict

def main():
    meshes = sys.argv[2]
    simplify = sys.argv[3]
    alreadyDone = glob.glob(meshes + "*.obj")
    alreadyDone = [os.path.basename(i)[:-4] for i in alreadyDone]

    labelsFolderPath = sys.argv[1]

    labelsPaths = sorted(glob.glob(labelsFolderPath +'*'))
    for ii,stack in enumerate(labelsPaths):
        if os.path.basename(stack) in alreadyDone:
            print("Detected already processed file. Skipping.")
            print("[Delete file in output folder to reprocess.]")
            continue
        print("Starting " + stack)
        labelStack = tifffile.imread(stack)
        
        #labelStack = np.dstack(labelStack)
        print("Loaded data stack " + str(ii) + "/" + str(len(labelsPaths)))
        print("Thresholding...")

        calcMesh(stack, labelStack, meshes, simplify)


if __name__ == "__main__":
    main()
