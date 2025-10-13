#include "principled_brdf_cuda.h"
#include <cuda_runtime.h>
#include <stdexcept>

// CUDA kernel
__global__ void add_cuda_kernel(const float* a, const float* b, float* c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// CUDA memory allocation
extern "C" void* cuda_allocate(size_t size) {
    void* ptr = nullptr;
    cudaError_t err = cudaMalloc(&ptr, size);
    if (err != cudaSuccess) {
        return nullptr;
    }
    return ptr;
}

// CUDA memory deallocation
extern "C" void cuda_free(void* ptr) {
    if (ptr) {
        cudaFree(ptr);
    }
}

// CUDA dummy add implementation
extern "C" void cuda_dummy_add(const float* a, const float* b, float* result, size_t n) {
    // Launch kernel
    add_cuda_kernel<<<(n + 255) / 256, 256>>>(a, b, result, n);
    
    // Synchronize and check for errors
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        // Clean up on error
        cudaFree(result);
        throw std::runtime_error("CUDA kernel execution failed");
    }
    
    // Check for kernel launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(result);
        throw std::runtime_error("CUDA kernel launch failed");
    }
}
