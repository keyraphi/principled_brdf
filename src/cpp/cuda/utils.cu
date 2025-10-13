#include "utils.h"
#include <stdexcept>

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
    float* data = static_cast<float*>(cuda_allocate(N * 3 * sizeof(float)));
    cuda_set_vec3(data, x, y, z, N);
    nb::capsule owner(data, [](void *p) noexcept { cuda_free(p); });
    return Vec3ArrayCUDA(data, {N, 3}, owner);
}

ScalarArrayCUDA create_default_scalar(size_t N, float value) {
    float* data = static_cast<float*>(cuda_allocate(N * sizeof(float)));
    cuda_set_scalar(data, value, N);
    nb::capsule owner(data, [](void *p) noexcept { cuda_free(p); });
    return ScalarArrayCUDA(data, {N}, owner);
}

ScalarArrayCUDA broadcast_scalar(const FlexScalarCUDA& source, size_t N) {
    if (source.shape(0) == N) {
        return ScalarArrayCUDA(source);
    }
    else if (source.shape(0) == 1) {
        float* data = static_cast<float*>(cuda_allocate(N * sizeof(float)));
        float value = source.data()[0];
        cuda_broadcast_scalar(data, value, N);
        nb::capsule owner(data, [](void *p) noexcept { cuda_free(p); });
        return ScalarArrayCUDA(data, {N}, owner);
    } else {
        throw std::runtime_error("Scalar parameter must have shape [1] or [N]");
    }
}

Vec3ArrayCUDA broadcast_vec3(const FlexVec3CUDA& source, size_t N) {
    if (source.shape(0) == N && source.shape(1) == 3) {
        return Vec3ArrayCUDA(source);
    }
    else if (source.shape(0) == 1 && source.shape(1) == 3) {
        float* data = static_cast<float*>(cuda_allocate(N * 3 * sizeof(float)));
        const float* src_data = source.data();
        cuda_broadcast_vec3(data, src_data, N);
        nb::capsule owner(data, [](void *p) noexcept { cuda_free(p); });
        return Vec3ArrayCUDA(data, {N, 3}, owner);
    } else {
        throw std::runtime_error("Vector parameter must have shape [1, 3] or [N, 3]");
    }
}

} // namespace cuda
