# setup.py
from distutils.core import Extension, setup

import numpy
from Cython.Build import cythonize

setup(
    ext_modules=cythonize(
        Extension(
            "boxes_from_map",
            sources=["boxes_from_map.pyx"],
            language="c",
            include_dirs=[numpy.get_include()],
            library_dirs=[],
            libraries=[],
            extra_compile_args=[],
            extra_link_args=[],
        )
    )
)
