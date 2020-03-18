from __future__ import print_function
from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("util.pyx")
    )

