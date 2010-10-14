from distutils.core import setup
import datetime
#version = datetime.date.today().isoformat().replace("-","")
version = "alpha"

packages = ["ivy", "ivy.vis"]

desc = "An interactive visual shell for phylogenetics"

setup(name="ivy",
      version=version,
      description=desc,
      long_description=file("README.txt").read(),
      author="Richard Ree",
      author_email="rree@fieldmuseum.org",
      url="http://www.reelab.net/ivy",
      license="LICENSE.txt",
      platforms="All",
      packages=packages,
      data_files=data_files,
      scripts=scripts)

