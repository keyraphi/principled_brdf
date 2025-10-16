#include "utils.h"
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <thrust/fill.h>
#include <thrust/device_ptr.h>
#include <stdexcept>
#include <vector_functions.h>


namespace cuda {
// CUDA memory allocation
void* cuda_allocate(size_t size) {
    void* ptr = nullptr;
    cudaError_t err = cudaMalloc(&ptr, size);
    if (err != cudaSuccess) {
        return nullptr;
    }
    return ptr;
}

// CUDA memory deallocation
void cuda_free(void* ptr) {
    if (ptr) {
        cudaFree(ptr);
    }
}

ScalarArrayCUDA broadcast_scalar(const FlexScalarCUDA& source, size_t N, float default_value) {
    if (source.shape(0) == N) {
        return ScalarArrayCUDA(source);
    }
    if(source.shape(0) == 1) {
      // copy default value to host for simple thrust::fill
        cudaError_t err = cudaMemcpy(&default_value, source.data(), sizeof(float), cudaMemcpyDeviceToHost);
        if(err != cudaSuccess) {
          throw ::std::runtime_error("Couldn't load source value from devce");
        }
    }
    float* data = static_cast<float*>(cuda_allocate(N * sizeof(float)));
    thrust::device_ptr<float> data_ptr(data);
    thrust::fill(data_ptr, data_ptr + N, default_value);
    nb::capsule owner(data, [](void *p) noexcept { cuda_free(p); });
    return ScalarArrayCUDA(data, {N}, owner);
}

Vec3ArrayCUDA broadcast_vec3(const FlexVec3CUDA& source, size_t N, float default_x, float default_y, float default_z) {
    if (source.shape(0) == N) {
        return Vec3ArrayCUDA{source};
    }
    float3 default_value{default_x, default_y, default_z};
    if (source.shape(0) == 1) {
        cudaError_t err = cudaMemcpy(&default_value, source.data(), sizeof(float3), cudaMemcpyDeviceToHost);
        if(err != cudaSuccess) {
          throw ::std::runtime_error("Couldn't load source value from devce");
        }
    }
        float3* data3 = static_cast<float3*>(cuda_allocate(N * sizeof(float3)));
        thrust::device_ptr<float3> data_ptr(data3);
        thrust::fill(data_ptr, data_ptr + N, default_value);
        float* data = reinterpret_cast<float*>(data3);
        nb::capsule owner(data, [](void *p) noexcept { cuda_free(p); });
        return Vec3ArrayCUDA(data, {N, 3}, owner);
}


} // namespace cuda
