import torch
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

from disney_brdf.core import dummy_add

def test_dummy_add_cpu():
    a = torch.ones((2, 3), dtype=torch.float32)
    b = torch.ones((2, 3), dtype=torch.float32) * 2
    c = dummy_add(a, b)
    expected = torch.ones((2, 3), dtype=torch.float32) * 3
    assert torch.allclose(c, expected), f"Expected {expected}, got {c}"

def test_dummy_add_cuda():
    if torch.cuda.is_available():
        a = torch.ones((2, 3), dtype=torch.float32, device='cuda')
        b = torch.ones((2, 3), dtype=torch.float32, device='cuda') * 2
        c = dummy_add(a, b)
        expected = torch.ones((2, 3), dtype=torch.float32, device='cuda') * 3
        assert torch.allclose(c, expected), f"Expected {expected}, got {c}"

def test_dummy_add_mixed_precision():
    # Test with different dtypes if needed
    a = torch.ones((2, 3), dtype=torch.float32)
    b = torch.ones((2, 3), dtype=torch.float32) * 2
    c = dummy_add(a, b)
    expected = torch.ones((2, 3), dtype=torch.float32) * 3
    assert torch.allclose(c, expected)

if __name__ == "__main__":
    test_dummy_add_cpu()
    print("CPU test passed!")
    
    if torch.cuda.is_available():
        test_dummy_add_cuda()
        print("CUDA test passed!")
    else:
        print("CUDA not available, skipping CUDA test")
    
    test_dummy_add_mixed_precision()
    print("Mixed precision test passed!")
