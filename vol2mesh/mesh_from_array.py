import tempfile
import subprocess
from io import BytesIO
from shutil import copyfileobj

import numpy as np
from skimage.measure import marching_cubes_lewiner

from .io_utils import TemporaryNamedPipe, AutoDeleteDir

def generate_obj(vertices_xyz, faces, normals_xyz=[]):
    """
    Given lists of vertices and faces, write them to a new BytesIO stream in .obj format.
    """
    mesh_bytes = BytesIO()
    mesh_bytes.write(b"# OBJ file\n")

    for (x,y,z) in vertices_xyz:
        mesh_bytes.write(f"v {x:.7f} {y:.7f} {z:.7f}\n".encode('utf-8'))

    for (x,y,z) in normals_xyz:
        mesh_bytes.write(f"vn {x:.7f} {y:.7f} {z:.7f}\n".encode('utf-8'))

    for (v1, v2, v3) in faces:
        mesh_bytes.write(f"f {v1} {v2} {v3} \n".encode('utf-8'))
    
    mesh_bytes.seek(0)
    return mesh_bytes


def mesh_from_array(volume_zyx, global_offset_zyx, downsample_factor=1, simplify_ratio=None, step_size=1, output_format='obj'):
    """
    Given a binary volume, convert it to a mesh in .obj format, optionally simplified.
    
    Parameters
    ----------
    
    volume_zyx:
        Binary volume (ZYX order)
    global_offset_zyx:
        Offset of the volume start corner in global non-downsampled coordinates: (z0,y0,x0)
    downsample_factor:
        Factor by which the given volume has been downsampled from its original size
    simplify_ratio:
        How much to simplify the generated mesh (or None to skip simplification)
    step_size:
        Passed to skimage.measure.marching_cubes_lewiner().
        Larger values result in coarser results via faster computation.
    output_format:
        Either 'drc' or 'obj'
    method:
        Either 'ilastik' or 'skimage'
    """
    assert output_format in ('obj', 'drc'), \
        f"Unknown output format: {output_format}.  Expected one of ('obj', 'drc')"

    if simplify_ratio == 1.0:
        simplify_ratio = None

    try:    
        vertices_zyx, faces, normals_zyx, _values = marching_cubes_lewiner(volume_zyx, 0.5, step_size=step_size)
    except ValueError:
        if volume_zyx.all():
            # To consider:
            # Alternatively, we could return an empty .obj file in this case,
            # but for now it seems better to force the caller to decide what she wants to do.
            raise ValueError("Can't create mesh from completely solid volume.\n"
                             "(Volume edges are not normally converted to mesh faces, so the mesh would be empty.)\n"
                             "Try padding the input with an empty halo.")
        raise

    # Rescale and translate
    vertices_zyx[:] *= downsample_factor
    vertices_zyx[:] += global_offset_zyx

    # OBJ format: XYZ order
    vertices_xyz = vertices_zyx[:, ::-1]
    normals_xyz = normals_zyx[:, ::-1]
    
    # OBJ format: Faces start at index 1 (not 0)
    faces += 1

    mesh_stream = generate_obj(vertices_xyz, faces, normals_xyz)
        
    child_processes = []

    try:
        # The fq-mesh-simplify tool rejects inputs that are too small (if the decimated face count would be less than 4).
        # We have to check for this in advance because we can't gracefully handle the error.
        # https://github.com/neurolabusc/Fast-Quadric-Mesh-Simplification-Pascal-/blob/master/c_code/Main.cpp
        if simplify_ratio is not None and len(faces) * simplify_ratio > 4:
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

