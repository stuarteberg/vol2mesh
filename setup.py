from setuptools import find_packages, setup

setup( name='vol2mesh',
       version='0.0',
       description='Tools for converting image volumes to meshes',
       url='https://github.com/mmorehea/vol2mesh',
       author='Michael Morehead',
       packages=find_packages(),
       package_data={},
       entry_points={
          'console_scripts': [
              'mesh_from_dvid_tarfile = vol2mesh.bin.mesh_from_dvid_tarfile:main',
              'sv_to_mesh = vol2mesh.bin.sv_to_mesh:main'
          ]
       }
     )
