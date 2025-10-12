#include <nanobind/nanobind.h>
#include <nanobind/tensor.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "cpu/disney_brdf_cpu.h"
#ifdef USE_CUDA
#include "cuda/disney_brdf_cuda.h"
#endif

namespace nb = nanobind;

// Dummy function for CPU tensors
nb::tensor<nb::numpy, float> dummy_add(const nb::tensor<nb::numpy, float> &a,
                                      const nb::tensor<nb::numpy, float> &b) {
    if (a.shape(0) != b.shape(0) || a.shape(1) != b.shape(1)) {
        throw std::runtime_error("Shape mismatch");
    }

    // Allocate output tensor with same shape
    size_t shape[2] = {a.shape(0), a.shape(1)};
    nb::tensor<nb::numpy, float> result(shape, 2);

    // Perform element-wise addition
    for (size_t i = 0; i < a.shape(0); ++i) {
        for (size_t j = 0; j < a.shape(1); ++j) {
            result(i, j) = a(i, j) + b(i, j);
        }
    }

    return result;
}

#ifdef USE_CUDA
// Dummy function for CUDA tensors  
nb::tensor<nb::cuda, float> dummy_add(const nb::tensor<nb::cuda, float> &a,
                                     const nb::tensor<nb::cuda, float> &b) {
    if (a.shape(0) != b.shape(0) || a.shape(1) != b.shape(1)) {
        throw std::runtime_error("Shape mismatch");
    }

    size_t shape[2] = {a.shape(0), a.shape(1)};
    nb::tensor<nb::cuda, float> result(shape, 2);

    // Get raw pointers
    const float *a_data = a.data();
    const float *b_data = b.data();
    float *result_data = result.data();

    // Simple CUDA kernel launch for element-wise addition
    size_t n = a.shape(0) * a.shape(1);
    add_cuda_kernel<<<(n + 255) / 256, 256>>>(a_data, b_data, result_data, n);
    cudaDeviceSynchronize();

    return result;
}
#endif

NB_MODULE(disney_brdf_core, m) {
    m.doc() = "Disney BRDF with automatic differentiation";
    
    // Bind both overloads to the same function name
    // nanobind will automatically dispatch based on tensor type
    m.def("dummy_add", &dummy_add, "Add two tensors (CPU version)");
    
    #ifdef USE_CUDA
    m.def("dummy_add", &dummy_add, "Add two tensors (CUDA version)");
    #endif
    
    // You can also add a version that accepts any device type for maximum flexibility
    m.def("dummy_add", [](const nb::tensor<> &a, const nb::tensor<> &b) -> nb::tensor<> {
        // This provides a fallback that works with any device type
        // but the specific overloads above will be preferred when types match exactly
        
        if (a.shape(0) != b.shape(0) || a.shape(1) != b.shape(1)) {
            throw std::runtime_error("Shape mismatch");
        }
        
        // Create output tensor on same device
        size_t shape[2] = {a.shape(0), a.shape(1)};
        auto result = nb::tensor<>(a.device_type(), a.dtype(), shape, 2);
        
        // For the generic version, we'll do a simple CPU fallback for now
        // In a real implementation, you'd dispatch to the appropriate backend
        if (a.device_type() == nb::device::cpu::id && b.device_type() == nb::device::cpu::id) {
            auto a_cpu = nb::tensor<nb::numpy, float>(a);
            auto b_cpu = nb::tensor<nb::numpy, float>(b);
            auto result_cpu = nb::tensor<nb::numpy, float>(result);
            
            for (size_t i = 0; i < a.shape(0); ++i) {
                for (size_t j = 0; j < a.shape(1); ++j) {
                    result_cpu(i, j) = a_cpu(i, j) + b_cpu(i, j);
                }
            }
        }
        #ifdef USE_CUDA
        else if (a.device_type() == nb::device::cuda::id && b.device_type() == nb::device::cuda::id) {
            auto a_cuda = nb::tensor<nb::cuda, float>(a);
            auto b_cuda = nb::tensor<nb::cuda, float>(b);
            auto result_cuda = nb::tensor<nb::cuda, float>(result);
            
            const float *a_data = a_cuda.data();
            const float *b_data = b_cuda.data();
            float *result_data = result_cuda.data();
            
            size_t n = a.shape(0) * a.shape(1);
            add_cuda_kernel<<<(n + 255) / 256, 256>>>(a_data, b_data, result_data, n);
            cudaDeviceSynchronize();
        }
        #endif
        else {
            throw std::runtime_error("Unsupported device type or device mismatch");
        }
        
        return result;
    }, "Add two tensors (generic version)");
}
