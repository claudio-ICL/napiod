import setuptools
import numpy

ext_modules = [
    setuptools.Extension('napiod.intensities',
                         sources=['napiod/intensities.c'],
                         extra_compile_args=["-ffast-math"],
                         )
]

setuptools.setup(
    name="napiod",
    ext_modules=ext_modules,
    include_dirs=[numpy.get_include()],
)
