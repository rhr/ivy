from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize

extensions = [
    Extension(
        'sse',
        ['sse.pyx'],
        libraries = ['gsl', 'blas']
        ),
]

setup(
  name = 'cython-experiments',
  ext_modules = extensions
)
