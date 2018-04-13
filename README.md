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
from vol2mesh import Mesh, mesh_from_array, concatenate_meshes

# Creation from binary vol (via marching cubes)
# For some binary ndarray 'binary_array'
box = [(0,0,0), (binary_vol.shape)]
mesh = Mesh.from_binary_vol( binary_vol, box )

# Creation from file(s)/buffer
mesh2 = Mesh.from_file( '/path/to/mesh.drc' )
mesh3 = Mesh.from_directory( '/path/to/meshes/' )
mesh4 = Mesh.from_bytes( obj_bytes, 'obj )

# Basic ops
mesh.laplacian_smooth(3)
mesh.simplify(0.2)
mesh.drop_normals() # If you don't want them in your output files

# Serialize to disk
mesh.serialize('/tmp/my-mesh.obj')
mesh.serialize('/tmp/my-mesh.drc')

# Serialize to buffer
mesh_bytes = mesh.serialize(fmt='drc')

# Combine meshes (with proper vertex renumbering in the faces)
combined_mesh = concatenate_meshes([mesh1, mesh2, mesh3])

# Optional: Deduplicate vertices
combined_mesh.stitch_adjacent_faces()

# Less common ops
mesh.drop_unused_vertices()
mesh.recompute_normals()


# Legacy API
mesh_bytes = mesh_from_array( binary_vol, box[0], 1, smoothing_rounds=3, simplify_ratio=0.2, output_format='drc' )

with open('my_mesh.drc', 'wb') as f:
    f.write(mesh_bytes)
```


Appendix: Dependencies
----------------------

```
conda install -c flyem-forge -c conda-forge fq-mesh-simplification draco
``` 

- Mesh decimation implementation: https://github.com/sp4cerat/Fast-Quadric-Mesh-Simplification
- Draco compressed format: https://github.com/google/draco
