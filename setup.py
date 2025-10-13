from setuptools import setup, find_packages

import subprocess
import sys
import os

def build_and_install_cpp_extension():
    """Build and install the C++ extension using CMake"""
    print("Building principled_brdf_functions C++ extension...")
    
    # Create build directory
    build_dir = "build_setup"
    os.makedirs(build_dir, exist_ok=True)
    
    try:
        # Configure with CMake
        subprocess.check_call([
            'cmake', '..', 
            f'-DPython_EXECUTABLE={sys.executable}',
            '-DCMAKE_BUILD_TYPE=Release'
        ], cwd=build_dir)
        
        # Build and install
        subprocess.check_call([
            'cmake', '--build', '.', '--target', 'principled_brdf_functions'
        ], cwd=build_dir)
        
        subprocess.check_call([
            'cmake', '--install', '.'
        ], cwd=build_dir)
        
        print("Successfully built and installed principled_brdf_functions")
        
    except subprocess.CalledProcessError as e:
        print(f"Failed to build C++ extension: {e}")
        print("Please ensure CMake and CUDA are installed")
        sys.exit(1)

# Build and install the C++ extension before setting up the Python package
build_and_install_cpp_extension()

setup(
    name="principled_brdf_torch",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch>=1.9.0",
    ],
    python_requires=">=3.7",
    author="Raphael Braun",
    author_email="keyraphi@gmail.com",
    description="PyTorch bindings for Principled BRDF with automatic differentiation",
)
