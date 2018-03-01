# vol2mesh

A mesh generating wrapper in Python 3, with optional simplification and draco compression.

Installation
------------

```bash
conda install -c flyem-forge vol2mesh
```

Python usage
------------

Example:

```python
from vol2mesh import Mesh, mesh_from_array

# For some binary ndarray 'binary_array'
box = [(0,0,0), (binary_vol.shape)]

# Standard API
mesh = Mesh.from_binary_vol( binary_vol, box )
mesh2 = Mesh.from_file( '/path/to/mesh.drc' )

mesh.laplacian_smooth(3, recompute_normals=False)
mesh.simplify(0.2, recompute_normals=False)
mesh.recompute_normals()
mesh_bytes = mesh.serialize(fmt='drc')

# Legacy API
mesh_bytes = mesh_from_array( binary_vol, box[0], 1, smoothing_rounds=3, simplify_ratio=0.2, output_format='drc' )

with open('my_mesh.drc', 'wb') as f:
    f.write(mesh_simple_compressed)
```

Command-line usage
------------------

Executables are installed to `$CONDA_PREFIX/bin`

Input is a list of folders containing 3D tiff stack. Output is a directory for meshes. Third option is decimation percentage.

Example:

```bash
vol2mesh ./data/ ./meshes/ .2
```


Appendix: Dependencies
----------------------

```
conda install -c flyem-forge -c conda-forge fq-mesh-simplification draco
``` 

- Mesh decimation implementation: https://github.com/sp4cerat/Fast-Quadric-Mesh-Simplification
- Draco compressed format: https://github.com/google/draco
