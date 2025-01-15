from setuptools import setup
from setuptools import find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="photoMD",
    version="1.0.0",
    author="Chen Zhou",
    author_email="chen.zhou@kit.edu",
    description="Excited states dynamics simulations with Parallel Active Learning (PAL).",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    install_requires=['numpy', "mpi4py", "importlib"],
    extras_require={
        "tf": ["tensorflow>=2.3.0"],
        "tf_gpu": ["tensorflow-gpu>=2.3.0"],
    },
    packages=find_packages(
        where='usr_example',
    ),
    include_package_data=False,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Software Development :: Libraries :: Python Modules"
    ],
    keywords=["materials", "science", "machine", "learning", "deep", "dynamics", "molecular", "potential", "active", "mpi", "parallel"]
)