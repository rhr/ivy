from distutils.core import setup
import datetime
version = datetime.date.today().isoformat().replace("-","")
#version = "0.2" # 2010-12-09

packages = [
    "ivy", "ivy.vis", "ivy.chars"
    ]
#packages=find_packages(exclude=["contrib","db"])

install_requires = [
    'ipython,'
    'numpy',
    'scipy',
    'matplotlib',
    'pillow',
    'biopython',
    'pyparsing',
    'lxml'
]

desc = "An interactive visual shell for phylogenetics"

package_data = {
    '': ["*.data", "*.txt", "*.nex"]
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
      install_requires=install_requires)

