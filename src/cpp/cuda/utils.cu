#include "utils.h"
#include <cstddef>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <nanobind/ndarray.h>
#include <stdexcept>
#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include <vector>
#include <vector_functions.h>

namespace nb = nanobind;

namespace cuda {
// CUDA memory allocation
void *cuda_allocate(size_t size) {
  void *ptr = nullptr;
  cudaError_t err = cudaMalloc(&ptr, size);
  if (err != cudaSuccess) {
    return nullptr;
  }
  return ptr;
}

// CUDA memory deallocation
void cuda_free(void *ptr) {
  if (ptr) {
    cudaFree(ptr);
  }
}

ScalarArrayCUDA broadcast_scalar(const FlexScalarCUDA &source, size_t N,
                                 float default_value) {
  if (source.shape(0) == N) {
    return ScalarArrayCUDA(source);
  }
  if (source.shape(0) == 1) {
    // copy default value to host for simple thrust::fill
    cudaError_t err = cudaMemcpy(&default_value, source.data(), sizeof(float),
                                 cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
      throw ::std::runtime_error("Couldn't load source value from device");
    }
  }
  float *data = static_cast<float *>(cuda_allocate(N * sizeof(float)));
  thrust::device_ptr<float> data_ptr(data);
  thrust::fill(data_ptr, data_ptr + N, default_value);
  nb::capsule owner(data, [](void *p) noexcept { cuda_free(p); });
  return ScalarArrayCUDA(data, {N}, owner);
}

Vec3ArrayCUDA broadcast_vec3(const FlexVec3CUDA &source, size_t N,
                             float default_x, float default_y,
                             float default_z) {
  if (source.shape(0) == N) {
    return Vec3ArrayCUDA{source};
  }
  float3 default_value{default_x, default_y, default_z};
  if (source.shape(0) == 1) {
    cudaError_t err = cudaMemcpy(&default_value, source.data(), sizeof(float3),
                                 cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
      throw ::std::runtime_error("Couldn't load source value from device");
    }
  }
  float3 *data3 = static_cast<float3 *>(cuda_allocate(N * sizeof(float3)));
  thrust::device_ptr<float3> data_ptr(data3);
  thrust::fill(data_ptr, data_ptr + N, default_value);
  float *data = reinterpret_cast<float *>(data3);
  nb::capsule owner(data, [](void *p) noexcept { cuda_free(p); });
  return Vec3ArrayCUDA(data, {N, 3}, owner);
}

// Helper to get CUDA device from nanobind ndarray
auto get_cuda_device_from_ndarray(const void *data_ptr) -> int {
  cudaPointerAttributes attributes;
  cudaError_t result = cudaPointerGetAttributes(&attributes, data_ptr);

  if (result != cudaSuccess) {
    throw ::std::runtime_error("Failed to get CUDA pointer attributes");
  }

  // attributes.device contains the device ID where the memory is allocated
  return attributes.device;
}

// Overload for multiple arrays - picks the first valid one
auto get_commond_cuda_device(const Vec3ArrayCUDA &omega_i, const Vec3ArrayCUDA &omega_o,
                     const FlexVec3CUDA &P_b, const FlexScalarCUDA &P_m,
                     const FlexScalarCUDA &P_ss, const FlexScalarCUDA &P_s,
                     const FlexScalarCUDA &P_r, const FlexScalarCUDA &P_st,
                     const FlexScalarCUDA &P_ani, const FlexScalarCUDA &P_sh,
                     const FlexScalarCUDA &P_sht, const FlexScalarCUDA &P_c,
                     const FlexScalarCUDA &P_cg, const FlexVec3CUDA &n)
    -> int {
  // Check all arrays and return the first valid device
  ::std::vector<int> devices;

  if (omega_i.size() > 0) {
    devices.push_back(cuda::get_cuda_device_from_ndarray(omega_i.data()));
  }
  if (omega_o.size() > 0) {
    devices.push_back(cuda::get_cuda_device_from_ndarray(omega_o.data()));
  }
  if (P_b.size() > 0) {
    devices.push_back(cuda::get_cuda_device_from_ndarray(P_b.data()));
  }
  if (P_m.size() > 0) {
    devices.push_back(cuda::get_cuda_device_from_ndarray(P_m.data()));
  }
  if (P_ss.size() > 0) {
    devices.push_back(cuda::get_cuda_device_from_ndarray(P_ss.data()));
  }
  if (P_s.size() > 0) {
    devices.push_back(cuda::get_cuda_device_from_ndarray(P_s.data()));
  }
  if (P_r.size() > 0) {
    devices.push_back(cuda::get_cuda_device_from_ndarray(P_r.data()));
  }
  if (P_st.size() > 0) {
    devices.push_back(cuda::get_cuda_device_from_ndarray(P_st.data()));
  }
  if (P_ani.size() > 0) {
    devices.push_back(cuda::get_cuda_device_from_ndarray(P_ani.data()));
  }
  if (P_sh.size() > 0) {
    devices.push_back(cuda::get_cuda_device_from_ndarray(P_sh.data()));
  }
  if (P_sht.size() > 0) {
    devices.push_back(cuda::get_cuda_device_from_ndarray(P_sht.data()));
  }
  if (P_c.size() > 0) {
    devices.push_back(cuda::get_cuda_device_from_ndarray(P_c.data()));
  }
  if (P_cg.size() > 0) {
    devices.push_back(cuda::get_cuda_device_from_ndarray(P_cg.data()));
  }
  if (n.size() > 0) {
    devices.push_back(cuda::get_cuda_device_from_ndarray(n.data()));
  }
  // If no arrays have data (shouldn't happen in practice), default to device
  // 0
  if (devices.empty()) {
    return 0;
  }
  int common_device = 0;
    // Verify all devices are the same
    common_device = devices[0];
    for (size_t i = 1; i < devices.size(); ++i) {
      if (devices[i] != common_device) {
        throw ::std::runtime_error("Input tensors are on different GPUs. All "
                                   "inputs must be on the same GPU.");
      }
    }
  return common_device;
}

} // namespace cuda
