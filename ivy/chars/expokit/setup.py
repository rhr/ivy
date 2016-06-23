from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Compiler.Options import directive_defaults
from Cython.Build import cythonize
import numpy

directive_defaults['linetrace'] = True
directive_defaults['binding'] = True

ext = Extension(
    "cyexpokit",
    ["cyexpokit.pyx"],
    libraries = ["lapack", "blas", "m"],
    extra_objects = ["dexpm_c.o", "expokit.o", "mataid.o", "clock.o"],
    define_macros=[('CYTHON_TRACE', '1')],
    include_dirs=[numpy.get_include()]
    )

if __name__ == "__main__":
    setup(name = "Expokit f->f90->c extension module",
          cmdclass = {"build_ext": build_ext},
          ext_modules = [ext], gdb_debug=True)
