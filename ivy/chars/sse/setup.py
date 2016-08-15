from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize
from Cython.Compiler.Options import directive_defaults

directive_defaults['linetrace'] = True
directive_defaults['binding'] = True
extensions = [
    Extension(
        'sse',
        ['sse.pyx'],
        libraries = ['gsl', 'blas'],
        define_macros=[('CYTHON_TRACE', '1')],
        ),
]

setup(
  name = 'cython-experiments',
  ext_modules = extensions
)
