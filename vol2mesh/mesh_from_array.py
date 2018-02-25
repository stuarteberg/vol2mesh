import tempfile
import subprocess
from io import BytesIO
from shutil import copyfileobj

import numpy as np

from .io_utils import TemporaryNamedPipe, AutoDeleteDir
from .obj_utils import write_obj, read_obj
from .normals import compute_vertex_normals
from .mesh_utils import binary_vol_to_mesh

def mesh_from_array(volume_zyx, global_offset_zyx, downsample_factor=1, simplify_ratio=None, output_format='obj', return_vertex_count=False):
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
    output_format:
        Either 'drc' or 'obj'
    method:
        Either 'ilastik' or 'skimage'
    return_vertex_count:
        If True, also return the APPROXIMATE vertex count
        (We don't count the vertexes after decimation; we assume that decimation
        was able to faithfully apply the requested simplify_ratio.)
    """
    assert output_format in ('obj', 'drc'), \
        f"Unknown output format: {output_format}.  Expected one of ('obj', 'drc')"

    box = [ global_offset_zyx,
            global_offset_zyx + downsample_factor * np.asarray(volume_zyx.shape) ]

    mesh = binary_vol_to_mesh(volume_zyx, box, 'skimage')
    vertices_zyx, faces, normals_zyx = mesh.vertices_zyx, mesh.faces, mesh.normals_zyx

    # Rescale and translate
    vertices_zyx[:] *= downsample_factor
    vertices_zyx[:] += global_offset_zyx

    mesh_stream = BytesIO()
    write_obj(vertices_zyx, faces, normals_zyx, mesh_stream)
    mesh_stream.seek(0)
    
    child_processes = []

    if simplify_ratio == 1.0:
        simplify_ratio = None

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
            
            # The fq-mesh-simplify tool does not compute normals.  Compute them.
            vertices_zyx, faces, _empty_normals = read_obj(mesh_stream)
            mesh_stream.close()

            normals_zyx = compute_vertex_normals(vertices_zyx, faces)
            
            mesh_stream = BytesIO()
            write_obj(vertices_zyx, faces, normals_zyx, mesh_stream)
            mesh_stream.seek(0)

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

        mesh_bytes = mesh_stream.read()
        
        # For now, this is hidden behind a flag for backwards compatibility
        if not return_vertex_count:
            return mesh_bytes

        return mesh_bytes, len(vertices_zyx)

    finally:
        # Explicitly wait() for the child processes
        # (avoids a warning from subprocess.Popen.__del__)
        for cmd, proc in child_processes:
            proc.wait(timeout=1.0)
            if proc.returncode != 0:
                raise RuntimeError(f"Child process returned an error code: {proc.returncode}.\n"
                                   f"Command was: {cmd}")

