from distutils.core import setup, Extension
import datetime
from distutils.command.build_py import build_py
from distutils.command.build_scripts import build_scripts
import subprocess


try:
    from Cython.Distutils import build_ext
except ImportError:
    use_cython = False
else:
    use_cython = True

version = datetime.date.today().isoformat().replace("-","")
#version = "0.2" # 2010-12-09
import sys
import os

# Cython extensions

cmdclass =  {'build_py': build_py,
             'build_scripts': build_scripts
             }
ext_modules = [ ]

if use_cython:
   ext_modules += [
       Extension("ivy.chars.expokit.cyexpokit", [ "ivy/chars/expokit/cyexpokit.pyx" ]),
   ]
   cmdclass.update({ 'build_ext': build_ext })
else:
   ext_modules += [
       Extension("ivy.chars.expokit.cyexpokit", [ "ivy/chars/expokit/cyexpokit.c" ])
   ]

packages = [
    "ivy", "ivy.vis", "ivy.chars", "ivy.chars.expokit", "ivy.sim", "ivy.chars.cy_tree","ivy.chars.sse"
    ]
#packages=find_packages(exclude=["contrib","db"])

#install_requires = ["numpy","scipy"]

desc = "An interactive visual shell for phylogenetics"

package_data = {
    '': ["*.data", "*.txt", "*.nex", "*"]
    }
setup(name="ivy-phylo",
      version=version,
      description=desc,
      long_description=open("README.rst").read(),
      author="Richard Ree",
      author_email="rree@fieldmuseum.org",
      url="http://www.reelab.net/ivy",
      license="GPL",
      platforms="All",
      packages=packages,
      package_data=package_data,
      cmdclass = cmdclass,
      ext_modules = ext_modules,
      classifiers=["Programming Language :: Python :: 2.7"])
