from distutils.core import setup
from Cython.Build import cythonize
import numpy
setup(
  name = 'cy_tree',
  ext_modules = cythonize("cy_tree.pyx"),
  include_dirs=[numpy.get_include()]
)
