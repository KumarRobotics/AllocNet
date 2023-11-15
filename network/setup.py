from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

class BinaryDistribution(build_ext):
    def run(self):
        pass

ext_modules = [
    Extension(
        "irispy",  # Replace with the desired module name
        sources=[],
        libraries=[],
    )
]

setup(
    name="irispy",  # Replace with the desired package name
    version="0.1",
    author="Xiatao Sun",
    author_email="sunxiatao@gmail.com",
    description="A Python wrapper for iris",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BinaryDistribution},
    packages=["irispy"],  # Replace with the desired package name
    package_data={"irispy": ["iris_wrapper.cpython-38-x86_64-linux-gnu.so"]},  # Replace with the actual .so file name
    zip_safe=False,
)