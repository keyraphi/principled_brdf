#include "utils.h"
#include <algorithm>
#include <cstddef>
#include <nanobind/nanobind.h>

namespace cpu {

auto broadcast_scalar(const FlexScalarCPU &source, size_t N,
                      float default_value) -> ScalarArrayCPU {
  if (source.shape(0) == N) {
    return ScalarArrayCPU{source};
  }
  if (source.shape(0) == 1) {
    default_value = source.data()[0];
  }
  auto *data = new float[N];
  std::fill(data, data + N, default_value);
  nb::capsule owner(data,
                    [](void *ptr) noexcept -> void { delete[] (float *)ptr; });
  return ScalarArrayCPU{data, {N}, owner};
}

auto broadcast_vec3(const FlexVec3CPU &source, size_t N, float default_x,
                    float default_y, float default_z) -> Vec3ArrayCPU {

  if (source.shape(0) == N) {
    return Vec3ArrayCPU{source};
  }
  if (source.shape(0) == 1) {
    const float *src_data = source.data();
    default_x = src_data[0];
    default_y = src_data[1];
    default_z = src_data[2];
  }
  auto *data = new float[N * 3];
  for (size_t i = 0; i < N; ++i) {
    data[(i * 3) + 0] = default_x;
    data[(i * 3) + 1] = default_y;
    data[(i * 3) + 2] = default_z;
  }
  nb::capsule owner(data, [](void *ptr) noexcept -> void { delete[] (float *)ptr; });
  return Vec3ArrayCPU(data, {N, 3}, owner);
}

} // namespace cpu
