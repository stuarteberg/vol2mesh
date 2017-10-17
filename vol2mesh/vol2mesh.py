from marching_cubes import march
import code
import tifffile
import numpy as np
import glob
from numpy import load
import subprocess
import threading
import os
import sys
import pickle
import platform
import tempfile
import threading
from contextlib import contextmanager
from functools import partial
from io import BytesIO


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

@contextmanager
def temp_pipe(name):
    """
    Context manager.
    Create a temporary named pipe, with the given basename (do not include directory).
    The pipe is deleted when the context is exited.
    
    yields: the full path to the named pipe
    """
    dir = tempfile.mkdtemp()
    path = f"{dir}/{name}"

    os.mkfifo(path)    
    yield path
    
    os.unlink(path)
    os.rmdir(dir)
    

def simplify_mesh(simplify_ratio, mesh_obj_text):
    """
    Simplify the given mesh (in .obj text format) using the fq-mesh-simplify
    command-line tool, but use named pipes instead of files (to avoid using the hard disk).
    
    simplify_ratio: float
    mesh_obj_text: bytes. The contents of an .obj file.
    """
    assert isinstance(mesh_obj_text, bytes), "Must give mesh_obj_text as bytes, not str"

    with temp_pipe('mesh.obj') as mesh_path, temp_pipe('simple.obj') as simple_path:

        def write_mesh():
            with open(mesh_path, 'wb') as f:
                f.write(mesh_obj_text)
        threading.Thread(target=write_mesh).start()
    
        cmd = f'fq-mesh-simplify "{mesh_path}" "{simple_path}" {simplify_ratio}'
        proc = subprocess.Popen(cmd, shell=True)

        try:
            with open(simple_path, 'rb') as f:
                return f.read()
        finally:
            proc.wait()
            if proc.returncode != 0:
                raise RuntimeError(f"fq-mesh-simplify returned an error code: {proc.returncode}")


def generate_obj(vertices_xyz, faces):
    """
    Given lists of vertices and faces, write them to a buffer in .obj format.
    """
    with BytesIO() as mesh_bytes:
        mesh_bytes.write(b"# OBJ file\n")
        for (x,y,z) in vertices_xyz:
            mesh_bytes.write(f"v {x:.7f} {y:.7f} {z:.7f}\n".encode('utf-8'))
        for (v1, v2, v3) in faces:
            mesh_bytes.write(f"f {v1} {v2} {v3} \n".encode('utf-8'))
        return mesh_bytes.getvalue()


def mesh_from_array(volume_zyx, box_zyx, downsample_factor, simplify_ratio=None):
    """
    Given a binary volume, convert it to a mesh in .obj format, optionally simplified.
    
    
    """
    volume_xyz = volume_zyx.transpose()
    box_xyz = box_zyx[:,::-1]

    vertices_xyz, normals, faces = march(volume_xyz, 3)  # 3 smoothing rounds

    # Rescale and translate
    vertices_xyz[:] *= downsample_factor
    vertices_xyz[:] += box_xyz[0]
    
    # I don't understand why we write face vertices in reverse order...
    # ...does marching_cubes give clockwise order instead of counter-clockwise?
    # Is it because we passed a fortran-order array?
    faces = faces[:, ::-1]
    faces += 1

    mesh_bytes = generate_obj(vertices_xyz, faces)
    
    if simplify_ratio is not None:
        mesh_bytes = simplify_mesh(simplify_ratio, mesh_bytes)

    return mesh_bytes
    

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
    #code.interact(local=locals())
    for ii,stack in enumerate(labelsPaths):
        if os.path.basename(stack) in alreadyDone:
            print("Detected already processed file. Skipping.")
            print("[Delete file in output folder to reprocess.]")
            continue
        print("Starting " + stack)
        labelStack = tifffile.imread(stack)
        #code.interact(local=locals())
        
        #labelStack = np.dstack(labelStack)
        print("Loaded data stack " + str(ii) + "/" + str(len(labelsPaths)))
        print("Thresholding...")

        calcMesh(stack, labelStack, meshes, simplify)


if __name__ == "__main__":
    main()
