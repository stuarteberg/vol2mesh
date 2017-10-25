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
              'slices2mesh = vol2mesh.slices2mesh:main',
              'slices2mesh1 = vol2mesh.slices2mesh1:main',
              'slices2mesh2 = vol2mesh.slices2mesh2:main',
              'vol2mesh = vol2mesh.vol2mesh:main',
              'vol2mesh4DVID = vol2mesh.vol2mesh4DVID:main'
          ]
       }
     )
