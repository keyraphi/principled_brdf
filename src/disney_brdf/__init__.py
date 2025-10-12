try:
    from .disney_brdf_core import *
except ImportError as e:
    print(f"Could not import Disney BRDF C++/CUDA extension: {e}")
    print("Please build the extension with: pip install -e .")

__version__ = "0.1.0"
