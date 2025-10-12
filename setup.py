from skbuild import setup
from setuptools import find_packages

setup(
    name="disney-brdf",
    version="0.1.0",
    description="Disney BRDF with derivatives as a custom PyTorch operation",
    author="Raphael Braun",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    cmake_install_dir="src/disney_brdf",
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy"
    ],
)
