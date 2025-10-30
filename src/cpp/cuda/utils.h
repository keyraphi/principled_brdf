#pragma once
#include <cmath>
#include <cstddef>
#include <cuda_runtime_api.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <optional>

namespace nb = nanobind;

namespace cuda {
// Forward declarations
using FlexScalarCUDA =
    nb::ndarray<const float, nb::shape<-1>, nb::c_contig, nb::device::cuda>;
using FlexVec3CUDA =
    nb::ndarray<const float, nb::shape<-1, 3>, nb::c_contig, nb::device::cuda>;
using ScalarArrayCUDA =
    nb::ndarray<const float, nb::shape<-1>, nb::c_contig, nb::device::cuda>;
using Vec3ArrayCUDA =
    nb::ndarray<const float, nb::shape<-1, 3>, nb::c_contig, nb::device::cuda>;

// CUDA memory management functions
void *cuda_allocate(size_t n);
void cuda_free(void *ptr);

// make sure the given array gets the full shape. If no source is given the
// default values are used
ScalarArrayCUDA broadcast_scalar(const FlexScalarCUDA &source, size_t N,
                                 float default_value);
Vec3ArrayCUDA broadcast_vec3(const FlexVec3CUDA &source, size_t N,
                             float default_x, float default_y, float default_z);

// Device handling utils
auto get_cuda_device_from_ndarray(const nb::ndarray<nb::device::cuda> &arr)
    -> int;
// set cuda device to common device of inputs
auto get_common_cuda_device(const Vec3ArrayCUDA &omega_i,
                            const Vec3ArrayCUDA &omega_o,
                            const ::std::optional<FlexVec3CUDA> &P_b,
                            const ::std::optional<FlexScalarCUDA> &P_m,
                            const ::std::optional<FlexScalarCUDA> &P_ss,
                            const ::std::optional<FlexScalarCUDA> &P_s,
                            const ::std::optional<FlexScalarCUDA> &P_r,
                            const ::std::optional<FlexScalarCUDA> &P_st,
                            const ::std::optional<FlexScalarCUDA> &P_ani,
                            const ::std::optional<FlexScalarCUDA> &P_sh,
                            const ::std::optional<FlexScalarCUDA> &P_sht,
                            const ::std::optional<FlexScalarCUDA> &P_c,
                            const ::std::optional<FlexScalarCUDA> &P_cg,
                            const ::std::optional<FlexVec3CUDA> &n) -> int;

class ScopedCudaDevice {
private:
  int original_device_;

public:
  ScopedCudaDevice(int new_device) {
    cudaGetDevice(&original_device_);
    cudaSetDevice(new_device);
  }

  ~ScopedCudaDevice() { cudaSetDevice(original_device_); }

  // Disallow copying
  ScopedCudaDevice(const ScopedCudaDevice &) = delete;
  ScopedCudaDevice &operator=(const ScopedCudaDevice &) = delete;
};

struct __attribute__((visibility("default"))) BRDFInputs {
  size_t N;

  Vec3ArrayCUDA omega_i;
  Vec3ArrayCUDA omega_o;
  Vec3ArrayCUDA P_b;
  ScalarArrayCUDA P_m;
  ScalarArrayCUDA P_ss;
  ScalarArrayCUDA P_s;
  ScalarArrayCUDA P_r;
  ScalarArrayCUDA P_st;
  ScalarArrayCUDA P_ani;
  ScalarArrayCUDA P_sh;
  ScalarArrayCUDA P_sht;
  ScalarArrayCUDA P_c;
  ScalarArrayCUDA P_cg;
  Vec3ArrayCUDA n;
};

struct Vec3 {
  float x, y, z;

  // Constructor from individual components
  __host__ __device__ Vec3(float x_, float y_, float z_)
      : x{x_}, y{y_}, z{z_} {}

  // Default constructor (initializes to zero)
  __host__ __device__ Vec3() = default;

  // Vector addition
  __host__ __device__ Vec3 operator+(const Vec3 &rhs) const {
    return Vec3(x + rhs.x, y + rhs.y, z + rhs.z);
  }
  __host__ __device__ Vec3 operator+(const float rhs) const {
    return Vec3(x + rhs, y + rhs, z + rhs);
  }
  __host__ __device__ Vec3 operator-(const Vec3 &rhs) const {
    return Vec3(x - rhs.x, y - rhs.y, z - rhs.z);
  }

  // Scalar multiplication (both directions)
  __host__ __device__ Vec3 operator*(float scalar) const {
    return Vec3(x * scalar, y * scalar, z * scalar);
  }
  __host__ __device__ friend Vec3 operator*(float scalar, const Vec3 &vec) {
    return vec * scalar;
  }

  // Dot product (vector multiplication)
  __host__ __device__ float operator*(const Vec3 &rhs) const {
    return x * rhs.x + y * rhs.y + z * rhs.z;
  }

  // Scalar division
  __host__ __device__ Vec3 operator/(float scalar) const {
    return Vec3(x / scalar, y / scalar, z / scalar);
  }

  __host__ __device__ Vec3 operator^(float exponent) const {
    return Vec3(powf(x, exponent), powf(y, exponent), powf(z, exponent));
  }

  // In-place normalization
  __host__ __device__ Vec3 &normalize() {
    float len = sqrtf(x * x + y * y + z * z);
    if (len > 0) {
      float inv_len = 1.f / len;
      x *= inv_len;
      y *= inv_len;
      z *= inv_len;
    }
    return *this;
  }

  // Element-wise Hadamard product
  __host__ __device__ Vec3 hadamard(const Vec3 &rhs) const {
    return Vec3(x * rhs.x, y * rhs.y, z * rhs.z);
  }
};

} // namespace cuda
