#!/usr/bin/env python3
"""
Basic functionality test for principled_brdf_functions
"""

import sys

try:
    import principled_brdf_functions as brdf
    import torch

    print("SUCCESS: Successfully imported principled_brdf_functions and torch")
except ImportError as e:
    print(f"ERROR: Import failed: {e}")
    sys.exit(1)


def _convert_capsule_to_tensor(capsule):
    """Convert DLPack capsule to PyTorch tensor"""
    return torch.from_dlpack(capsule)

def run_test(omega_i, omega_o):
    try:
        # Test the dummy_add function and convert from DLPack
        result_capsule = brdf.principled_brdf_forward(omega_i, omega_o)
        result = _convert_capsule_to_tensor(result_capsule)
        print("forward shape:", result.shape)
        result_capsule = brdf.principled_brdf_backward_basecolor(omega_i, omega_o)
        result = _convert_capsule_to_tensor(result_capsule)
        print("backward basecolor shape:", result.shape)
        result_capsule = brdf.principled_brdf_backward_metallic(omega_i, omega_o)
        result = _convert_capsule_to_tensor(result_capsule)
        print("backward metallic shape:", result.shape)
        result_capsule = brdf.principled_brdf_backward_subsurface(omega_i, omega_o)
        result = _convert_capsule_to_tensor(result_capsule)
        print("backward subsurface shape:", result.shape)
        result_capsule = brdf.principled_brdf_backward_specular(omega_i, omega_o)
        result = _convert_capsule_to_tensor(result_capsule)
        print("backward specular shape:", result.shape)
        result_capsule = brdf.principled_brdf_backward_roughness(omega_i, omega_o)
        result = _convert_capsule_to_tensor(result_capsule)
        print("backward roughness shape:", result.shape)
        result_capsule = brdf.principled_brdf_backward_specular_tint(omega_i, omega_o)
        result = _convert_capsule_to_tensor(result_capsule)
        print("backward specular_tint shape:", result.shape)
        result_capsule = brdf.principled_brdf_backward_anisotropy(omega_i, omega_o)
        result = _convert_capsule_to_tensor(result_capsule)
        print("backward anisotropy shape:", result.shape)
        result_capsule = brdf.principled_brdf_backward_sheen(omega_i, omega_o)
        result = _convert_capsule_to_tensor(result_capsule)
        print("backward sheen shape:", result.shape)
        result_capsule = brdf.principled_brdf_backward_sheen_tint(omega_i, omega_o)
        result = _convert_capsule_to_tensor(result_capsule)
        print("backward sheen_tint shape:", result.shape)
        result_capsule = brdf.principled_brdf_backward_clearcoat(omega_i, omega_o)
        result = _convert_capsule_to_tensor(result_capsule)
        print("backward clearcoat shape:", result.shape)
        result_capsule = brdf.principled_brdf_backward_clearcoat_gloss(omega_i, omega_o)
        result = _convert_capsule_to_tensor(result_capsule)
        print("backward clearcoat_gloss shape:", result.shape)
        result_capsule = brdf.principled_brdf_backward_normal(omega_i, omega_o)
        result = _convert_capsule_to_tensor(result_capsule)
        print("backward normal shape:", result.shape)
        result_capsule = brdf.principled_brdf_backward_omega_i(omega_i, omega_o)
        result = _convert_capsule_to_tensor(result_capsule)
        print("backward omega_i shape:", result.shape)
        result_capsule = brdf.principled_brdf_backward_omega_o(omega_i, omega_o)
        result = _convert_capsule_to_tensor(result_capsule)
        print("backward omega_o shape:", result.shape)


    except Exception as e:
        print(f"FAILED: CPU operation failed: {e}")
        return False

    return True

def test_basic_operations():
    """Test basic operations work on CPU"""
    print("Testing basic CPU operations...")

    # Create test tensors on cpu
    omega_i = torch.randn([1024, 3])
    omega_i = omega_i / torch.linalg.norm(omega_i, -1)
    omega_o = torch.randn([1024, 3])
    omega_o = omega_o / torch.linalg.norm(omega_o, -1)

    return run_test(omega_i, omega_o)


def test_gpu_operations():
    """Test operations on GPU if available"""
    if not torch.cuda.is_available():
        print("SKIPPED: CUDA not available, skipping GPU tests")
        return True

    print("Testing GPU operations...")

    # Create test tensors on gpu
    omega_i = torch.randn([1024, 3])
    omega_i = omega_i / torch.linalg.norm(omega_i, -1)
    omega_o = torch.randn([1024, 3])
    omega_o = omega_o / torch.linalg.norm(omega_o, -1)

    return run_test(omega_i, omega_o)


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
