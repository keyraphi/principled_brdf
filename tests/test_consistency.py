#!/usr/bin/env python3
"""
Consistency test between CPU, GPU, and PyTorch implementations
"""

import sys

try:
    import principled_brdf_functions
    import torch

    print("SUCCESS: Successfully imported modules")
except ImportError as e:
    print(f"ERROR: Import failed: {e}")
    sys.exit(1)


def _convert_capsule_to_tensor(capsule):
    """Convert DLPack capsule to PyTorch tensor"""
    return torch.from_dlpack(capsule)


def test_cpu_gpu_consistency():
    """Test that CPU and GPU implementations give same results"""
    print("Testing CPU-GPU consistency...")

    if not torch.cuda.is_available():
        print("SKIPPED: CUDA not available, skipping CPU-GPU consistency test")
        return True

    try:
        # Create test data
        a_cpu = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        b_cpu = torch.tensor([[5.0, 6.0], [7.0, 8.0]])

        # Move to GPU
        a_gpu = a_cpu.cuda()
        b_gpu = b_cpu.cuda()

        # Compute on CPU and convert from DLPack
        result_capsule_cpu = principled_brdf_functions.dummy_add(a_cpu, b_cpu)
        result_cpu = _convert_capsule_to_tensor(result_capsule_cpu)

        # Compute on GPU and convert from DLPack
        result_capsule_gpu = principled_brdf_functions.dummy_add(a_gpu, b_gpu)
        result_gpu = _convert_capsule_to_tensor(result_capsule_gpu)

        # Move GPU result back to CPU for comparison
        result_gpu_cpu = result_gpu.cpu()

        # Check consistency
        if torch.allclose(result_cpu, result_gpu_cpu, atol=1e-6):
            print("SUCCESS: CPU and GPU implementations are consistent")
            return True
        else:
            print(
                f"FAILED: CPU-GPU inconsistency: CPU {result_cpu}, GPU {result_gpu_cpu}"
            )
            return False

    except Exception as e:
        print(f"FAILED: CPU-GPU consistency test failed: {e}")
        return False


def test_pytorch_consistency():
    """Test that our implementation matches PyTorch's results"""
    print("Testing PyTorch consistency...")

    try:
        # Create test data
        a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        b = torch.tensor([[5.0, 6.0], [7.0, 8.0]])

        # Our implementation and convert from DLPack
        result_capsule = principled_brdf_functions.dummy_add(a, b)
        our_result = _convert_capsule_to_tensor(result_capsule)

        # PyTorch reference
        pytorch_result = a + b

        # Check consistency
        if torch.allclose(our_result, pytorch_result, atol=1e-6):
            print("SUCCESS: Our implementation matches PyTorch reference")
            return True
        else:
            print(
                f"FAILED: PyTorch inconsistency: ours {our_result}, PyTorch {pytorch_result}"
            )
            return False

    except Exception as e:
        print(f"FAILED: PyTorch consistency test failed: {e}")
        return False


if __name__ == "__main__":
    print("Running consistency tests...")
    print("=" * 50)

    success = True
    success &= test_cpu_gpu_consistency()
    success &= test_pytorch_consistency()

    print("=" * 50)
    if success:
        print("SUCCESS: All consistency tests passed!")
        sys.exit(0)
    else:
        print("FAILED: Some consistency tests failed!")
        sys.exit(1)
