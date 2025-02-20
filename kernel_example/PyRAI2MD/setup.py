######################################################
#
# PyRAI2MD 2 setup file for compling fssh cython library
#
# Author Jingbai Li
# Oct 11 2021
#
######################################################

from setuptools import setup
from setuptools import find_packages
from Cython.Build import cythonize
import numpy as np

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="PyRAI2MD",
    version="2.1.0",
    author="Jingbai Li",
    description="Python Rapid Artificial Intelligence Ab Initio Molecular Dynamics.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=['numpy', "Cython", "mpi4py"],
    packages=find_packages(where=["."], exclude=["TEST*"]),
    include_package_data=False,
    ext_modules = cythonize("./PyRAI2MD/Dynamics/Propagators/fssh.pyx",compiler_directives={'language_level' : "3"}),
    include_dirs=[np.get_include()],
    package_dir={'cython_fssh': ''},
)