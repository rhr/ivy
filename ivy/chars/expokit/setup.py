from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext


ext = Extension(
    "cyexpokit",
    ["cyexpokit.pyx"],
    libraries = ["lapack", "blas", "m"],
    extra_objects = ["dexpm_c.o", "expokit.o", "mataid.o", "clock.o"]
    )

if __name__ == "__main__":
    setup(name = "Expokit f->f90->c extension module",
          cmdclass = {"build_ext": build_ext},
          ext_modules = [ext])
      
