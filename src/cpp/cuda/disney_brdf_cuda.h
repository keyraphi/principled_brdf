#pragma once

#include <cuda_runtime.h>

// CUDA kernel declarations
__global__ void add_cuda_kernel(const float* a, const float* b, float* c, size_t n);

void disney_brdf_forward_cuda(const float* input, float* output, int64_t size);
void disney_brdf_backward_cuda(const float* input, const float* grad_output, 
                              float* grad_input, int64_t size);
