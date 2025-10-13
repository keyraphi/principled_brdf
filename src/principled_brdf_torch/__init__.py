"""
Principled BRDF - A physically-based bidirectional reflectance distribution function
implementation with automatic differentiation support.
"""

# Import the C++ extension from the same directory
from principled_brdf_functions import *
from .principled_brdf import PrincipledBRDFFunction, PrincipledBRDF, dummy_add

__all__ = ['PrincipledBRDFFunction', 'PrincipledBRDF', 'dummy_add']
