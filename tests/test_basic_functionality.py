#!/usr/bin/env python3
"""
Basic functionality test for principled_brdf_functions
"""

import sys

try:
    import principled_brdf_functions
    import torch

    print("SUCCESS: Successfully imported principled_brdf_functions and torch")
except ImportError as e:
    print(f"ERROR: Import failed: {e}")
    sys.exit(1)


def _convert_capsule_to_tensor(capsule):
    """Convert DLPack capsule to PyTorch tensor"""
    return torch.from_dlpack(capsule)


def test_basic_operations():
    """Test basic operations work on CPU"""
    print("Testing basic CPU operations...")

    # Create test tensors
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    b = torch.tensor([[5.0, 6.0], [7.0, 8.0]])

    try:
        # Test the dummy_add function and convert from DLPack
        result_capsule = principled_brdf_functions.dummy_add(a, b)
        result = _convert_capsule_to_tensor(result_capsule)
        expected = a + b

        # Check if results match
        if torch.allclose(result, expected):
            print("SUCCESS: CPU dummy_add works correctly")
            return True
        else:
            print(f"FAILED: CPU dummy_add failed: expected {expected}, got {result}")
            return False

    except Exception as e:
        print(f"FAILED: CPU operation failed: {e}")
        return False


def test_gpu_operations():
    """Test operations on GPU if available"""
    if not torch.cuda.is_available():
        print("SKIPPED: CUDA not available, skipping GPU tests")
        return True

    print("Testing GPU operations...")

    try:
        # Create test tensors on GPU
        a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device="cuda")
        b = torch.tensor([[5.0, 6.0], [7.0, 8.0]], device="cuda")

        # Test the dummy_add function on GPU and convert from DLPack
        result_capsule = principled_brdf_functions.dummy_add(a, b)
        result = _convert_capsule_to_tensor(result_capsule)
        expected = a + b

        # Check if results match
        if torch.allclose(result, expected):
            print("SUCCESS: GPU dummy_add works correctly")
            return True
        else:
            print(f"FAILED: GPU dummy_add failed: expected {expected}, got {result}")
            return False

    except Exception as e:
        print(f"FAILED: GPU operation failed: {e}")
        return False


if __name__ == "__main__":
    print("Running basic functionality tests...")
    print("=" * 50)

    success = True
    success &= test_basic_operations()
    success &= test_gpu_operations()

    print("=" * 50)
    if success:
        print("SUCCESS: All basic functionality tests passed!")
        sys.exit(0)
    else:
        print("FAILED: Some tests failed!")
        sys.exit(1)
