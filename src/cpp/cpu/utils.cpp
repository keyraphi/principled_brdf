#include "utils.h"
#include <stdexcept>

namespace cpu {

Vec3ArrayCPU create_default_vec3(size_t N, float x, float y, float z) {
    float* data = new float[N * 3];
    for (size_t i = 0; i < N; ++i) {
        data[i * 3 + 0] = x;
        data[i * 3 + 1] = y;
        data[i * 3 + 2] = z;
    }
    nb::capsule owner(data, [](void *p) noexcept { delete[] (float*)p; });
    return Vec3ArrayCPU(data, {N, 3}, owner);
}

ScalarArrayCPU create_default_scalar(size_t N, float value) {
    float* data = new float[N];
    std::fill(data, data + N, value);
    nb::capsule owner(data, [](void *p) noexcept { delete[] (float*)p; });
    return ScalarArrayCPU(data, {N}, owner);
}

ScalarArrayCPU broadcast_scalar(const FlexScalarCPU& source, size_t N) {
    if (source.shape(0) == N) {
        return ScalarArrayCPU(source);
    }
    else if (source.shape(0) == 1) {
        float* data = new float[N];
        float value = source.data()[0];
        std::fill(data, data + N, value);
        nb::capsule owner(data, [](void *p) noexcept { delete[] (float*)p; });
        return ScalarArrayCPU(data, {N}, owner);
    } else {
        throw std::runtime_error("Scalar parameter must have shape [1] or [N]");
    }
}

Vec3ArrayCPU broadcast_vec3(const FlexVec3CPU& source, size_t N) {
    if (source.shape(0) == N && source.shape(1) == 3) {
        return Vec3ArrayCPU(source);
    }
    else if (source.shape(0) == 1 && source.shape(1) == 3) {
        float* data = new float[N * 3];
        const float* src_data = source.data();
        for (size_t i = 0; i < N; ++i) {
            data[i * 3 + 0] = src_data[0];
            data[i * 3 + 1] = src_data[1];
            data[i * 3 + 2] = src_data[2];
        }
        nb::capsule owner(data, [](void *p) noexcept { delete[] (float*)p; });
        return Vec3ArrayCPU(data, {N, 3}, owner);
    } else {
        throw std::runtime_error("Vector parameter must have shape [1, 3] or [N, 3]");
    }
}

} // namespace cpu
