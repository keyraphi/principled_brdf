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

Vec3ArrayCUDA create_default_vec3(size_t N, float x, float y, float z) {
    float3* data3 = static_cast<float3*>(cuda_allocate(N * 3 * sizeof(float)));
    thrust::device_ptr<float3> data_ptr(data3);
    thrust::fill(data_ptr, data_ptr+N, make_float3(x, y, z));
    float* data = reinterpret_cast<float*>(data3);
    nb::capsule owner(data, [](void *p) noexcept { cuda_free(p); });
    return Vec3ArrayCUDA(data, {N, 3}, owner);
}

ScalarArrayCUDA create_default_scalar(size_t N, float value) {
    float* data = static_cast<float*>(cuda_allocate(N * sizeof(float)));
    thrust::device_ptr<float> data_ptr(data);
    thrust::fill(data_ptr, data_ptr + N, value);
    nb::capsule owner(data, [](void *p) noexcept { cuda_free(p); });
    return ScalarArrayCUDA(data, {N}, owner);
}

ScalarArrayCUDA broadcast_scalar(const FlexScalarCUDA& source, size_t N) {
    if (source.shape(0) == N) {
        return ScalarArrayCUDA(source);
    }
    else if (source.shape(0) == 1) {
        float* data = static_cast<float*>(cuda_allocate(N * sizeof(float)));
        float value;
        cudaError_t err = cudaMemcpy(&value, source.data(), sizeof(float), cudaMemcpyDeviceToHost);
        if(err != cudaSuccess) {
          throw ::std::runtime_error("Couldn't load source value from devce");
        }
        thrust::device_ptr<float> data_ptr(data);
        thrust::fill(data_ptr, data_ptr + N, value);
        nb::capsule owner(data, [](void *p) noexcept { cuda_free(p); });
        return ScalarArrayCUDA(data, {N}, owner);
    } else {
      throw ::std::runtime_error("Scalar parameter must have shape [1] or [N]");
    }
}

Vec3ArrayCUDA broadcast_vec3(const FlexVec3CUDA& source, size_t N) {
    if (source.shape(0) == N && source.shape(1) == 3) {
        return Vec3ArrayCUDA(source);
    }
    else if (source.shape(0) == 1 && source.shape(1) == 3) {
        float3* data3 = static_cast<float3*>(cuda_allocate(N * sizeof(float3)));

        float3 value;
        cudaError_t err = cudaMemcpy(&value, source.data(), sizeof(float3), cudaMemcpyDeviceToHost);
        if(err != cudaSuccess) {
          throw ::std::runtime_error("Couldn't load source value from devce");
        }
        thrust::device_ptr<float3> data_ptr(data3);
        thrust::fill(data_ptr, data_ptr + N, value);
        float* data = reinterpret_cast<float*>(data3);
        nb::capsule owner(data, [](void *p) noexcept { cuda_free(p); });
        return Vec3ArrayCUDA(data, {N, 3}, owner);
    } else {
      throw ::std::runtime_error("Vector parameter must have shape [1, 3] or [N, 3]");
    }
}


} // namespace cuda
