import torch
import numpy as np

try:
    from . import disney_brdf_core
except ImportError:
    disney_brdf_core = None

class DisneyBRDF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor):
        if disney_brdf_core is None:
            raise RuntimeError("Disney BRDF C++ extension not available")
        
        # Convert to numpy/cuda tensors for nanobind
        if input_tensor.is_cuda:
            output = disney_brdf_core.disney_brdf_forward_cuda(input_tensor)
        else:
            output = disney_brdf_core.disney_brdf_forward_cpu(input_tensor)
        
        ctx.save_for_backward(input_tensor)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        if disney_brdf_core is None:
            raise runtime_error("Disney BRDF C++ extension not available")
        
        input_tensor, = ctx.saved_tensors
        
        if grad_output.is_cuda:
            grad_input = disney_brdf_core.disney_brdf_backward_cuda(input_tensor, grad_output)
        else:
            grad_input = disney_brdf_core.disney_brdf_backward_cpu(input_tensor, grad_output)
        
        return grad_input

def disney_brdf(input_tensor):
    return DisneyBRDF.apply(input_tensor)

# Updated: Use the unified dummy_add function
def dummy_add(a, b):
    """Test function that uses our C++/CUDA implementation with automatic device dispatch"""
    if disney_brdf_core is None:
        raise RuntimeError("Disney BRDF C++ extension not available")
    return disney_brdf_core.dummy_add(a, b)
