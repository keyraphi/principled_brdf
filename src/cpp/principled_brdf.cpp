#include <cstddef>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <optional>
#include <stdexcept>

#include "cpu/principled_brdf_cpu.h"
#include "cpu/utils.h"
#include "cuda/principled_brdf_cuda.h"
#include "cuda/utils.h"
namespace nb = nanobind;
using namespace nb::literals;

// Device-specific type aliases with exact constraints
using Vec3ArrayCPU =
    nb::ndarray<const float, nb::shape<-1, 3>, nb::c_contig, nb::device::cpu>;
using ScalarArrayCPU =
    nb::ndarray<const float, nb::shape<-1>, nb::c_contig, nb::device::cpu>;
using OutputArrayCPU =
    nb::ndarray<float, nb::shape<-1, 3>, nb::c_contig, nb::device::cpu>;
using JacobiOutputArrayCPU =
    nb::ndarray<float, nb::shape<-1, 3, 3>, nb::c_contig, nb::device::cpu>;

using Vec3ArrayCUDA =
    nb::ndarray<const float, nb::shape<-1, 3>, nb::c_contig, nb::device::cuda>;
using ScalarArrayCUDA =
    nb::ndarray<const float, nb::shape<-1>, nb::c_contig, nb::device::cuda>;
using OutputArrayCUDA =
    nb::ndarray<float, nb::shape<-1, 3>, nb::c_contig, nb::device::cuda>;
using JacobiOutputArrayCUDA =
    nb::ndarray<float, nb::shape<-1, 3, 3>, nb::c_contig, nb::device::cuda>;

// Flexible input types that allow [1] or [N]
using FlexScalarCPU =
    nb::ndarray<const float, nb::shape<-1>, nb::c_contig, nb::device::cpu>;
using FlexVec3CPU =
    nb::ndarray<const float, nb::shape<-1, 3>, nb::c_contig, nb::device::cpu>;

using FlexScalarCUDA =
    nb::ndarray<const float, nb::shape<-1>, nb::c_contig, nb::device::cuda>;
using FlexVec3CUDA =
    nb::ndarray<const float, nb::shape<-1, 3>, nb::c_contig, nb::device::cuda>;

auto inputs_with_defaults(const Vec3ArrayCUDA &omega_i,
                          const Vec3ArrayCUDA &omega_o,
                          const std::optional<FlexVec3CUDA> &P_b,
                          const std::optional<FlexScalarCUDA> &P_m,
                          const std::optional<FlexScalarCUDA> &P_ss,
                          const std::optional<FlexScalarCUDA> &P_s,
                          const std::optional<FlexScalarCUDA> &P_r,
                          const std::optional<FlexScalarCUDA> &P_st,
                          const std::optional<FlexScalarCUDA> &P_ani,
                          const std::optional<FlexScalarCUDA> &P_sh,
                          const std::optional<FlexScalarCUDA> &P_sht,
                          const std::optional<FlexScalarCUDA> &P_c,
                          const std::optional<FlexScalarCUDA> &P_cg,
                          const std::optional<FlexVec3CUDA> &n)
    -> cuda::BRDFInputs {

  cuda::BRDFInputs params;
  params.N = omega_i.shape(0);
  params.omega_i = Vec3ArrayCUDA(omega_i);
  params.omega_o = Vec3ArrayCUDA(omega_o);

  if (P_b.has_value()) {
    params.P_b = cuda::broadcast_vec3(*P_b, params.N, 0.8F, 0.8F, 0.8F);
  }
  if (P_m.has_value()) {
    params.P_m = cuda::broadcast_scalar(*P_m, params.N, 0.0F);
  }
  if (P_ss.has_value()) {
    params.P_ss = cuda::broadcast_scalar(*P_ss, params.N, 0.0F);
  }
  if (P_s.has_value()) {
    params.P_s = cuda::broadcast_scalar(*P_s, params.N, 0.5F);
  }
  if (P_r.has_value()) {
    params.P_r = cuda::broadcast_scalar(*P_r, params.N, 0.5F);
  }
  if (P_st.has_value()) {
    params.P_st = cuda::broadcast_scalar(*P_st, params.N, 0.0F);
  }
  if (P_ani.has_value()) {
    params.P_ani = cuda::broadcast_scalar(*P_ani, params.N, 0.0F);
  }
  if (P_sh.has_value()) {
    params.P_sh = cuda::broadcast_scalar(*P_sh, params.N, 0.0F);
  }
  if (P_sht.has_value()) {
    params.P_sht = cuda::broadcast_scalar(*P_sht, params.N, 0.5F);
  }
  if (P_c.has_value()) {
    params.P_c = cuda::broadcast_scalar(*P_c, params.N, 0.0F);
  }
  if (P_cg.has_value()) {
    params.P_cg = cuda::broadcast_scalar(*P_cg, params.N, 1.0F);
  }
  if (n.has_value()) {
    params.n = cuda::broadcast_vec3(*n, params.N, 0.0F, 0.0F, 1.0F);
  }

  return params;
}

auto inputs_with_defaults(const Vec3ArrayCPU &omega_i,
                          const Vec3ArrayCPU &omega_o,
                          const std::optional<FlexVec3CPU> &P_b,
                          const std::optional<FlexScalarCPU> &P_m,
                          const std::optional<FlexScalarCPU> &P_ss,
                          const std::optional<FlexScalarCPU> &P_s,
                          const std::optional<FlexScalarCPU> &P_r,
                          const std::optional<FlexScalarCPU> &P_st,
                          const std::optional<FlexScalarCPU> &P_ani,
                          const std::optional<FlexScalarCPU> &P_sh,
                          const std::optional<FlexScalarCPU> &P_sht,
                          const std::optional<FlexScalarCPU> &P_c,
                          const std::optional<FlexScalarCPU> &P_cg,
                          const std::optional<FlexVec3CPU> &n)
    -> cpu::BRDFInputs {

  cpu::BRDFInputs params;
  params.N = omega_i.shape(0);
  params.omega_i = Vec3ArrayCPU(omega_i);
  params.omega_o = Vec3ArrayCPU(omega_o);
  if (P_b.has_value()) {
    params.P_b = cpu::broadcast_vec3(*P_b, params.N, 0.8F, 0.8F, 0.8F);
  }
  if (P_m.has_value()) {
    params.P_m = cpu::broadcast_scalar(*P_m, params.N, 0.0F);
  }
  if (P_ss.has_value()) {
    params.P_ss = cpu::broadcast_scalar(*P_ss, params.N, 0.0F);
  }
  if (P_s.has_value()) {
    params.P_s = cpu::broadcast_scalar(*P_s, params.N, 0.5F);
  }
  if (P_r.has_value()) {
    params.P_r = cpu::broadcast_scalar(*P_r, params.N, 0.5F);
  }
  if (P_st.has_value()) {
    params.P_st = cpu::broadcast_scalar(*P_st, params.N, 0.0F);
  }
  if (P_ani.has_value()) {
    params.P_ani = cpu::broadcast_scalar(*P_ani, params.N, 0.0F);
  }
  if (P_sh.has_value()) {
    params.P_sh = cpu::broadcast_scalar(*P_sh, params.N, 0.0F);
  }
  if (P_sht.has_value()) {
    params.P_sht = cpu::broadcast_scalar(*P_sht, params.N, 0.5F);
  }
  if (P_c.has_value()) {
    params.P_c = cpu::broadcast_scalar(*P_c, params.N, 0.0F);
  }
  if (P_c.has_value()) {
    params.P_cg = cpu::broadcast_scalar(*P_cg, params.N, 1.0F);
  }
  if (n.has_value()) {
    params.n = cpu::broadcast_vec3(*n, params.N, 0.0F, 0.0F, 1.0F);
  }

  return params;
}

// FORWARD /////////////////////////////////////////////////////////////////////
auto principled_brdf_forward_cpu(const Vec3ArrayCPU &omega_i,
                                 const Vec3ArrayCPU &omega_o,
                                 const FlexVec3CPU &P_b = FlexVec3CPU(),
                                 const FlexScalarCPU &P_m = FlexScalarCPU(),
                                 const FlexScalarCPU &P_ss = FlexScalarCPU(),
                                 const FlexScalarCPU &P_s = FlexScalarCPU(),
                                 const FlexScalarCPU &P_r = FlexScalarCPU(),
                                 const FlexScalarCPU &P_st = FlexScalarCPU(),
                                 const FlexScalarCPU &P_ani = FlexScalarCPU(),
                                 const FlexScalarCPU &P_sh = FlexScalarCPU(),
                                 const FlexScalarCPU &P_sht = FlexScalarCPU(),
                                 const FlexScalarCPU &P_c = FlexScalarCPU(),
                                 const FlexScalarCPU &P_cg = FlexScalarCPU(),
                                 const FlexVec3CPU &n = FlexVec3CPU())
    -> OutputArrayCPU {
  // Compolete inputs with defaults and broadcast constants for all required
  // parameters
  cpu::BRDFInputs params =
      inputs_with_defaults(omega_i, omega_o, P_b, P_m, P_ss, P_s, P_r, P_st,
                           P_ani, P_sh, P_sht, P_c, P_cg, n);

  auto *result_data = new float[params.N * 3];
  principled_brdf_forward_cpu_impl(
      params.omega_i.data(), params.omega_o.data(), params.P_b.data(),
      params.P_m.data(), params.P_ss.data(), params.P_s.data(),
      params.P_r.data(), params.P_st.data(), params.P_ani.data(),
      params.P_sh.data(), params.P_sht.data(), params.P_c.data(),
      params.P_cg.data(), params.n.data(), result_data, params.N);

  nb::capsule owner(result_data,
                    [](void *ptr) noexcept -> void { delete[] (float *)ptr; });
  return OutputArrayCPU(result_data, {params.N, 3}, owner);
}

auto principled_brdf_forward_cuda(
    const Vec3ArrayCUDA &omega_i, const Vec3ArrayCUDA &omega_o,
    const FlexVec3CUDA &P_b = FlexVec3CUDA(),
    const FlexScalarCUDA &P_m = FlexScalarCUDA(),
    const FlexScalarCUDA &P_ss = FlexScalarCUDA(),
    const FlexScalarCUDA &P_s = FlexScalarCUDA(),
    const FlexScalarCUDA &P_r = FlexScalarCUDA(),
    const FlexScalarCUDA &P_st = FlexScalarCUDA(),
    const FlexScalarCUDA &P_ani = FlexScalarCUDA(),
    const FlexScalarCUDA &P_sh = FlexScalarCUDA(),
    const FlexScalarCUDA &P_sht = FlexScalarCUDA(),
    const FlexScalarCUDA &P_c = FlexScalarCUDA(),
    const FlexScalarCUDA &P_cg = FlexScalarCUDA(),
    const FlexVec3CUDA &n = FlexVec3CUDA()) -> OutputArrayCUDA {

  // Ensure that correct gpu is used for computation
  int target_device =
      cuda::get_common_cuda_device(omega_i, omega_o, P_b, P_m, P_ss, P_s, P_r,
                                   P_st, P_ani, P_sh, P_sht, P_c, P_cg, n);
  cuda::ScopedCudaDevice device(target_device);

  // Compolete inputs with defaults and broadcast constants for all required
  // parameters
  cuda::BRDFInputs params =
      inputs_with_defaults(omega_i, omega_o, P_b, P_m, P_ss, P_s, P_r, P_st,
                           P_ani, P_sh, P_sht, P_c, P_cg, n);

  auto *result_data =
      static_cast<float *>(cuda::cuda_allocate(params.N * 3 * sizeof(float)));
  if (result_data == nullptr) {
    throw std::runtime_error("Failed to allocate CUDA memory for output");
  }
  principled_brdf_forward_cuda_impl(
      params.omega_i.data(), params.omega_o.data(), params.P_b.data(),
      params.P_m.data(), params.P_ss.data(), params.P_s.data(),
      params.P_r.data(), params.P_st.data(), params.P_ani.data(),
      params.P_sh.data(), params.P_sht.data(), params.P_c.data(),
      params.P_cg.data(), params.n.data(), result_data, params.N);

  nb::capsule owner(result_data,
                    [](void *ptr) noexcept -> void { cuda::cuda_free(ptr); });
  return OutputArrayCUDA(result_data, {params.N, 3}, owner);
}

// BACKWARD ////////////////////////////////////////////////////////////////////

// PARTIAL DERIVATIVE W.R.T P_c ////////////////
auto principled_brdf_backward_P_b_cpu(
    const Vec3ArrayCPU &omega_i, const Vec3ArrayCPU &omega_o,
    const FlexVec3CPU &P_b = FlexVec3CPU(),
    const FlexScalarCPU &P_m = FlexScalarCPU(),
    const FlexScalarCPU &P_ss = FlexScalarCPU(),
    const FlexScalarCPU &P_s = FlexScalarCPU(),
    const FlexScalarCPU &P_r = FlexScalarCPU(),
    const FlexScalarCPU &P_st = FlexScalarCPU(),
    const FlexScalarCPU &P_ani = FlexScalarCPU(),
    const FlexScalarCPU &P_sh = FlexScalarCPU(),
    const FlexScalarCPU &P_sht = FlexScalarCPU(),
    const FlexVec3CPU &n = FlexVec3CPU()) -> JacobiOutputArrayCPU {

  // Compolete inputs with defaults and broadcast constants for all required
  // parameters
  cpu::BRDFInputs params =
      inputs_with_defaults(omega_i, omega_o, P_b, P_m, P_ss, P_s, P_r, P_st,
                           P_ani, P_sh, P_sht, std::nullopt, std::nullopt, n);

  // Result are N 3x3 Jacobians
  auto *result_data = new float[params.N * 9];
  principled_brdf_backward_P_b_cpu_impl(
      params.omega_i.data(), params.omega_o.data(), params.P_b.data(),
      params.P_m.data(), params.P_ss.data(), params.P_s.data(),
      params.P_r.data(), params.P_st.data(), params.P_ani.data(),
      params.P_sh.data(), params.P_sht.data(), params.n.data(), result_data,
      params.N);

  nb::capsule owner(result_data,
                    [](void *ptr) noexcept -> void { delete[] (float *)ptr; });
  return JacobiOutputArrayCPU(result_data, {params.N, 3, 3}, owner);
}

// CUDA version with broadcasting and defaults
auto principled_brdf_backward_P_b_cuda(
    const Vec3ArrayCUDA &omega_i, const Vec3ArrayCUDA &omega_o,
    const FlexVec3CUDA &P_b = FlexVec3CUDA(),
    const FlexScalarCUDA &P_m = FlexScalarCUDA(),
    const FlexScalarCUDA &P_ss = FlexScalarCUDA(),
    const FlexScalarCUDA &P_s = FlexScalarCUDA(),
    const FlexScalarCUDA &P_r = FlexScalarCUDA(),
    const FlexScalarCUDA &P_st = FlexScalarCUDA(),
    const FlexScalarCUDA &P_ani = FlexScalarCUDA(),
    const FlexScalarCUDA &P_sh = FlexScalarCUDA(),
    const FlexScalarCUDA &P_sht = FlexScalarCUDA(),
    const FlexVec3CUDA &n = FlexVec3CUDA()) -> JacobiOutputArrayCUDA {

  // First ensure that correct gpu is used for all allocations and computations
  int target_device = cuda::get_common_cuda_device(
      omega_i, omega_o, P_b, P_m, P_ss, P_s, P_r, P_st, P_ani, P_sh, P_sht,
      std::nullopt, std::nullopt, n);
  cuda::ScopedCudaDevice device(target_device);

  // Compolete inputs with defaults and broadcast constant inputs for all
  // required parameters
  cuda::BRDFInputs params =
      inputs_with_defaults(omega_i, omega_o, P_b, P_m, P_ss, P_s, P_r, P_st,
                           P_ani, P_sh, P_sht, std::nullopt, std::nullopt, n);

  // Result are N 3x3 Jacobians
  auto *result_data =
      static_cast<float *>(cuda::cuda_allocate(params.N * 9 * sizeof(float)));
  if (result_data == nullptr) {
    throw std::runtime_error("Failed to allocate CUDA memory for output");
  }
  principled_brdf_backward_P_b_cuda_impl(
      params.omega_i.data(), params.omega_o.data(), params.P_b.data(),
      params.P_m.data(), params.P_ss.data(), params.P_s.data(),
      params.P_r.data(), params.P_st.data(), params.P_ani.data(),
      params.P_sh.data(), params.P_sht.data(), params.n.data(), result_data,
      params.N);

  nb::capsule owner(result_data,
                    [](void *ptr) noexcept -> void { cuda::cuda_free(ptr); });
  return JacobiOutputArrayCUDA(result_data, {params.N, 3, 3}, owner);
}

// PARTIAL DERIVATIVE w.r.t P_m ////////////////////////////////////////////////
auto principled_brdf_backward_P_m_cpu(
    const Vec3ArrayCPU &omega_i, const Vec3ArrayCPU &omega_o,
    const FlexVec3CPU &P_b = FlexVec3CPU(),
    const FlexScalarCPU &P_ss = FlexScalarCPU(),
    const FlexScalarCPU &P_s = FlexScalarCPU(),
    const FlexScalarCPU &P_r = FlexScalarCPU(),
    const FlexScalarCPU &P_st = FlexScalarCPU(),
    const FlexScalarCPU &P_ani = FlexScalarCPU(),
    const FlexScalarCPU &P_sh = FlexScalarCPU(),
    const FlexScalarCPU &P_sht = FlexScalarCPU(),
    const FlexVec3CPU &n = FlexVec3CPU()) -> OutputArrayCPU {
  // Compolete inputs with defaults and broadcast constants for all required
  // parameters
  cpu::BRDFInputs params = inputs_with_defaults(
      omega_i, omega_o, P_b, std::nullopt, P_ss, P_s, P_r, P_st, P_ani, P_sh,
      P_sht, std::nullopt, std::nullopt, n);

  // each result is a partial derivative of the r,g,b brdf
  auto *result_data = new float[params.N * 3];
  principled_brdf_backward_P_m_cpu_impl(
      params.omega_i.data(), params.omega_o.data(), params.P_b.data(),
      params.P_ss.data(), params.P_s.data(), params.P_r.data(),
      params.P_st.data(), params.P_ani.data(), params.P_sh.data(),
      params.P_sht.data(), params.n.data(), result_data, params.N);

  nb::capsule owner(result_data,
                    [](void *ptr) noexcept -> void { delete[] (float *)ptr; });
  return OutputArrayCPU(result_data, {params.N, 3}, owner);
}

auto principled_brdf_backward_P_m_cuda(
    const Vec3ArrayCUDA &omega_i, const Vec3ArrayCUDA &omega_o,
    const FlexVec3CUDA &P_b = FlexVec3CUDA(),
    const FlexScalarCUDA &P_ss = FlexScalarCUDA(),
    const FlexScalarCUDA &P_s = FlexScalarCUDA(),
    const FlexScalarCUDA &P_r = FlexScalarCUDA(),
    const FlexScalarCUDA &P_st = FlexScalarCUDA(),
    const FlexScalarCUDA &P_ani = FlexScalarCUDA(),
    const FlexScalarCUDA &P_sh = FlexScalarCUDA(),
    const FlexScalarCUDA &P_sht = FlexScalarCUDA(),
    const FlexVec3CUDA &n = FlexVec3CUDA()) -> OutputArrayCUDA {

  // Ensure that correct gpu is used for computation
  int target_device = cuda::get_common_cuda_device(
      omega_i, omega_o, P_b, std::nullopt, P_ss, P_s, P_r, P_st, P_ani, P_sh,
      P_sht, std::nullopt, std::nullopt, n);
  cuda::ScopedCudaDevice device(target_device);

  // Compolete inputs with defaults and broadcast constants for all required
  // parameters
  cuda::BRDFInputs params = inputs_with_defaults(
      omega_i, omega_o, P_b, std::nullopt, P_ss, P_s, P_r, P_st, P_ani, P_sh,
      P_sht, std::nullopt, std::nullopt, n);

  // each result is a partial derivative of the r,g,b brdf
  auto *result_data =
      static_cast<float *>(cuda::cuda_allocate(params.N * 3 * sizeof(float)));
  if (result_data == nullptr) {
    throw std::runtime_error("Failed to allocate CUDA memory for output");
  }
  principled_brdf_backward_P_m_cuda_impl(
      params.omega_i.data(), params.omega_o.data(), params.P_b.data(),
      params.P_ss.data(), params.P_s.data(), params.P_r.data(),
      params.P_st.data(), params.P_ani.data(), params.P_sh.data(),
      params.P_sht.data(), params.n.data(), result_data, params.N);

  nb::capsule owner(result_data,
                    [](void *ptr) noexcept -> void { cuda::cuda_free(ptr); });
  return OutputArrayCUDA(result_data, {params.N, 3}, owner);
}

// PARTIAL DERIVATIVE w.r.t P_ss //////////////////////////////////////////////
auto principled_brdf_backward_P_ss_cpu(
    const Vec3ArrayCPU &omega_i, const Vec3ArrayCPU &omega_o,
    const FlexVec3CPU &P_b = FlexVec3CPU(),
    const FlexScalarCPU &P_m = FlexScalarCPU(),
    const FlexScalarCPU &P_r = FlexScalarCPU(),
    const FlexVec3CPU &n = FlexVec3CPU()) -> OutputArrayCPU {
  // Compolete inputs with defaults and broadcast constants for all required
  // parameters
  cpu::BRDFInputs params = inputs_with_defaults(
      omega_i, omega_o, P_b, P_m, std::nullopt, std::nullopt, P_r, std::nullopt,
      std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, n);

  // each result is a partial derivative of the r,g,b brdf
  auto *result_data = new float[params.N * 3];
  principled_brdf_backward_P_ss_cpu_impl(
      params.omega_i.data(), params.omega_o.data(), params.P_b.data(),
      params.P_m.data(), params.P_r.data(), params.n.data(), result_data,
      params.N);

  nb::capsule owner(result_data,
                    [](void *ptr) noexcept -> void { delete[] (float *)ptr; });
  return OutputArrayCPU(result_data, {params.N, 3}, owner);
}

auto principled_brdf_backward_P_ss_cuda(
    const Vec3ArrayCUDA &omega_i, const Vec3ArrayCUDA &omega_o,
    const FlexVec3CUDA &P_b = FlexVec3CUDA(),
    const FlexScalarCUDA &P_m = FlexScalarCUDA(),
    const FlexScalarCUDA &P_r = FlexScalarCUDA(),
    const FlexVec3CUDA &n = FlexVec3CUDA()) -> OutputArrayCUDA {

  // Ensure that correct gpu is used for computation
  int target_device = cuda::get_common_cuda_device(
      omega_i, omega_o, P_b, P_m, std::nullopt, std::nullopt, P_r, std::nullopt,
      std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, n);
  cuda::ScopedCudaDevice device(target_device);

  // Compolete inputs with defaults and broadcast constants for all required
  // parameters
  cuda::BRDFInputs params = inputs_with_defaults(
      omega_i, omega_o, P_b, P_m, std::nullopt, std::nullopt, P_r, std::nullopt,
      std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, n);

  // each result is a partial derivative of the r,g,b brdf
  auto *result_data =
      static_cast<float *>(cuda::cuda_allocate(params.N * 3 * sizeof(float)));
  if (result_data == nullptr) {
    throw std::runtime_error("Failed to allocate CUDA memory for output");
  }
  principled_brdf_backward_P_ss_cuda_impl(
      params.omega_i.data(), params.omega_o.data(), params.P_b.data(),
      params.P_m.data(), params.P_r.data(), params.n.data(), result_data,
      params.N);

  nb::capsule owner(result_data,
                    [](void *ptr) noexcept -> void { cuda::cuda_free(ptr); });
  return OutputArrayCUDA(result_data, {params.N, 3}, owner);
}

// PARTIAL DERIVATIVE w.r.t P_s //////////////////////////////////////////////
auto principled_brdf_backward_P_s_cpu(
    const Vec3ArrayCPU &omega_i, const Vec3ArrayCPU &omega_o,
    const FlexVec3CPU &P_b = FlexVec3CPU(),
    const FlexScalarCPU &P_m = FlexScalarCPU(),
    const FlexScalarCPU &P_st = FlexScalarCPU(),
    const FlexVec3CPU &n = FlexVec3CPU()) -> OutputArrayCPU {
  // Compolete inputs with defaults and broadcast constants for all required
  // parameters
  cpu::BRDFInputs params = inputs_with_defaults(
      omega_i, omega_o, P_b, P_m, std::nullopt, std::nullopt, std::nullopt,
      P_st, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
      std::nullopt, n);

  auto *result_data = new float[params.N * 3];
  principled_brdf_backward_P_s_cpu_impl(
      params.omega_i.data(), params.omega_o.data(), params.P_b.data(),
      params.P_m.data(), params.P_st.data(), params.n.data(), result_data,
      params.N);

  nb::capsule owner(result_data,
                    [](void *ptr) noexcept -> void { delete[] (float *)ptr; });
  return OutputArrayCPU(result_data, {params.N, 3}, owner);
}

// CUDA version with broadcasting and defaults
auto principled_brdf_backward_P_s_cuda(
    const Vec3ArrayCUDA &omega_i, const Vec3ArrayCUDA &omega_o,
    const FlexVec3CUDA &P_b = FlexVec3CUDA(),
    const FlexScalarCUDA &P_m = FlexScalarCUDA(),
    const FlexScalarCUDA &P_st = FlexScalarCUDA(),
    const FlexVec3CUDA &n = FlexVec3CUDA()) -> OutputArrayCUDA {

  // Ensure that correct gpu is used for computation
  int target_device = cuda::get_common_cuda_device(
      omega_i, omega_o, P_b, P_m, std::nullopt, std::nullopt, std::nullopt,
      P_st, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
      std::nullopt, n);
  cuda::ScopedCudaDevice device(target_device);

  // Compolete inputs with defaults and broadcast constants for all required
  // parameters
  cuda::BRDFInputs params = inputs_with_defaults(
      omega_i, omega_o, P_b, P_m, std::nullopt, std::nullopt, std::nullopt,
      P_st, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
      std::nullopt, n);

  auto *result_data =
      static_cast<float *>(cuda::cuda_allocate(params.N * 3 * sizeof(float)));
  if (result_data == nullptr) {
    throw std::runtime_error("Failed to allocate CUDA memory for output");
  }
  principled_brdf_backward_P_s_cuda_impl(
      params.omega_i.data(), params.omega_o.data(), params.P_b.data(),
      params.P_m.data(), params.P_st.data(), params.n.data(), result_data,
      params.N);

  nb::capsule owner(result_data,
                    [](void *ptr) noexcept -> void { cuda::cuda_free(ptr); });
  return OutputArrayCUDA(result_data, {params.N, 3}, owner);
}

// PARTIAL DERIVATIVE w.r.t P_r //////////////////////////////////////////////
auto principled_brdf_backward_P_r_cpu(
    const Vec3ArrayCPU &omega_i, const Vec3ArrayCPU &omega_o,
    const FlexVec3CPU &P_b = FlexVec3CPU(),
    const FlexScalarCPU &P_m = FlexScalarCPU(),
    const FlexScalarCPU &P_ss = FlexScalarCPU(),
    const FlexScalarCPU &P_s = FlexScalarCPU(),
    const FlexScalarCPU &P_r = FlexScalarCPU(),
    const FlexScalarCPU &P_st = FlexScalarCPU(),
    const FlexScalarCPU &P_ani = FlexScalarCPU(),
    const FlexVec3CPU &n = FlexVec3CPU()) -> OutputArrayCPU {
  // Compolete inputs with defaults and broadcast constants for all required
  // parameters
  cpu::BRDFInputs params = inputs_with_defaults(
      omega_i, omega_o, P_b, P_m, P_ss, P_s, P_r, P_st, P_ani, std::nullopt,
      std::nullopt, std::nullopt, std::nullopt, n);

  auto *result_data = new float[params.N * 3];
  principled_brdf_backward_P_r_cpu_impl(
      params.omega_i.data(), params.omega_o.data(), params.P_b.data(),
      params.P_m.data(), params.P_ss.data(), params.P_s.data(),
      params.P_r.data(), params.P_st.data(), params.P_ani.data(),
      params.n.data(), result_data, params.N);

  nb::capsule owner(result_data,
                    [](void *ptr) noexcept -> void { delete[] (float *)ptr; });
  return OutputArrayCPU(result_data, {params.N, 3}, owner);
}

auto principled_brdf_backward_P_r_cuda(
    const Vec3ArrayCUDA &omega_i, const Vec3ArrayCUDA &omega_o,
    const FlexVec3CUDA &P_b = FlexVec3CUDA(),
    const FlexScalarCUDA &P_m = FlexScalarCUDA(),
    const FlexScalarCUDA &P_ss = FlexScalarCUDA(),
    const FlexScalarCUDA &P_s = FlexScalarCUDA(),
    const FlexScalarCUDA &P_r = FlexScalarCUDA(),
    const FlexScalarCUDA &P_st = FlexScalarCUDA(),
    const FlexScalarCUDA &P_ani = FlexScalarCUDA(),
    const FlexVec3CUDA &n = FlexVec3CUDA()) -> OutputArrayCUDA {

  // Ensure that correct gpu is used for computation
  int target_device = cuda::get_common_cuda_device(
      omega_i, omega_o, P_b, P_m, P_ss, P_s, P_r, P_st, P_ani, std::nullopt,
      std::nullopt, std::nullopt, std::nullopt, n);
  cuda::ScopedCudaDevice device(target_device);

  // Compolete inputs with defaults and broadcast constants for all required
  // parameters
  cuda::BRDFInputs params = inputs_with_defaults(
      omega_i, omega_o, P_b, P_m, P_ss, P_s, P_r, P_st, P_ani, std::nullopt,
      std::nullopt, std::nullopt, std::nullopt, n);

  auto *result_data =
      static_cast<float *>(cuda::cuda_allocate(params.N * 3 * sizeof(float)));
  if (result_data == nullptr) {
    throw std::runtime_error("Failed to allocate CUDA memory for output");
  }
  principled_brdf_backward_P_r_cuda_impl(
      params.omega_i.data(), params.omega_o.data(), params.P_b.data(),
      params.P_m.data(), params.P_ss.data(), params.P_s.data(),
      params.P_r.data(), params.P_st.data(), params.P_ani.data(),
      params.n.data(), result_data, params.N);

  nb::capsule owner(result_data,
                    [](void *ptr) noexcept -> void { cuda::cuda_free(ptr); });
  return OutputArrayCUDA(result_data, {params.N, 3}, owner);
}

auto principled_brdf_backward_P_st_cpu(
    const Vec3ArrayCPU &omega_i, const Vec3ArrayCPU &omega_o,
    const FlexVec3CPU &P_b = FlexVec3CPU(),
    const FlexScalarCPU &P_m = FlexScalarCPU(),
    const FlexScalarCPU &P_s = FlexScalarCPU(),
    const FlexScalarCPU &P_r = FlexScalarCPU(),
    const FlexScalarCPU &P_ani = FlexScalarCPU(),
    const FlexVec3CPU &n = FlexVec3CPU()) -> OutputArrayCPU {
  // Compolete inputs with defaults and broadcast constants for all required
  // parameters
  cpu::BRDFInputs params = inputs_with_defaults(
      omega_i, omega_o, P_b, P_m, std::nullopt, P_s, P_r, std::nullopt, P_ani,
      std::nullopt, std::nullopt, std::nullopt, std::nullopt, n);

  auto *result_data = new float[params.N * 3];
  principled_brdf_backward_P_st_cpu_impl(
      params.omega_i.data(), params.omega_o.data(), params.P_b.data(),
      params.P_m.data(), params.P_s.data(), params.P_r.data(),
      params.P_ani.data(), params.n.data(), result_data, params.N);

  nb::capsule owner(result_data,
                    [](void *ptr) noexcept -> void { delete[] (float *)ptr; });
  return OutputArrayCPU(result_data, {params.N, 3}, owner);
}

auto principled_brdf_backward_P_st_cuda(
    const Vec3ArrayCUDA &omega_i, const Vec3ArrayCUDA &omega_o,
    const FlexVec3CUDA &P_b = FlexVec3CUDA(),
    const FlexScalarCUDA &P_m = FlexScalarCUDA(),
    const FlexScalarCUDA &P_s = FlexScalarCUDA(),
    const FlexScalarCUDA &P_r = FlexScalarCUDA(),
    const FlexScalarCUDA &P_ani = FlexScalarCUDA(),
    const FlexVec3CUDA &n = FlexVec3CUDA()) -> OutputArrayCUDA {

  // Ensure that correct gpu is used for computation
  int target_device = cuda::get_common_cuda_device(
      omega_i, omega_o, P_b, P_m, std::nullopt, P_s, P_r, std::nullopt, P_ani,
      std::nullopt, std::nullopt, std::nullopt, std::nullopt, n);
  cuda::ScopedCudaDevice device(target_device);

  // Compolete inputs with defaults and broadcast constants for all required
  // parameters
  cuda::BRDFInputs params = inputs_with_defaults(
      omega_i, omega_o, P_b, P_m, std::nullopt, P_s, P_r, std::nullopt, P_ani,
      std::nullopt, std::nullopt, std::nullopt, std::nullopt, n);

  auto *result_data =
      static_cast<float *>(cuda::cuda_allocate(params.N * 3 * sizeof(float)));
  if (result_data == nullptr) {
    throw std::runtime_error("Failed to allocate CUDA memory for output");
  }
  principled_brdf_backward_P_st_cuda_impl(
      params.omega_i.data(), params.omega_o.data(), params.P_b.data(),
      params.P_m.data(), params.P_s.data(), params.P_r.data(),
      params.P_ani.data(), params.n.data(), result_data, params.N);

  nb::capsule owner(result_data,
                    [](void *ptr) noexcept -> void { cuda::cuda_free(ptr); });
  return OutputArrayCUDA(result_data, {params.N, 3}, owner);
}

auto principled_brdf_backward_P_ani_cpu(
    const Vec3ArrayCPU &omega_i, const Vec3ArrayCPU &omega_o,
    const FlexVec3CPU &P_b = FlexVec3CPU(),
    const FlexScalarCPU &P_m = FlexScalarCPU(),
    const FlexScalarCPU &P_s = FlexScalarCPU(),
    const FlexScalarCPU &P_r = FlexScalarCPU(),
    const FlexScalarCPU &P_st = FlexScalarCPU(),
    const FlexScalarCPU &P_ani = FlexScalarCPU(),
    const FlexVec3CPU &n = FlexVec3CPU()) -> OutputArrayCPU {
  // Compolete inputs with defaults and broadcast constants for all required
  // parameters
  cpu::BRDFInputs params = inputs_with_defaults(
      omega_i, omega_o, P_b, P_m, std::nullopt, P_s, P_r, P_st, P_ani,
      std::nullopt, std::nullopt, std::nullopt, std::nullopt, n);

  auto *result_data = new float[params.N * 3];
  principled_brdf_backward_P_ani_cpu_impl(
      params.omega_i.data(), params.omega_o.data(), params.P_b.data(),
      params.P_m.data(), params.P_s.data(), params.P_r.data(),
      params.P_st.data(), params.P_ani.data(), params.n.data(), result_data,
      params.N);

  nb::capsule owner(result_data,
                    [](void *ptr) noexcept -> void { delete[] (float *)ptr; });
  return OutputArrayCPU(result_data, {params.N, 3}, owner);
}

auto principled_brdf_backward_P_ani_cuda(
    const Vec3ArrayCUDA &omega_i, const Vec3ArrayCUDA &omega_o,
    const FlexVec3CUDA &P_b = FlexVec3CUDA(),
    const FlexScalarCUDA &P_m = FlexScalarCUDA(),
    const FlexScalarCUDA &P_s = FlexScalarCUDA(),
    const FlexScalarCUDA &P_r = FlexScalarCUDA(),
    const FlexScalarCUDA &P_st = FlexScalarCUDA(),
    const FlexScalarCUDA &P_ani = FlexScalarCUDA(),
    const FlexVec3CUDA &n = FlexVec3CUDA()) -> OutputArrayCUDA {

  // Ensure that correct gpu is used for computation
  int target_device = cuda::get_common_cuda_device(
      omega_i, omega_o, P_b, P_m, std::nullopt, P_s, P_r, P_st, P_ani,
      std::nullopt, std::nullopt, std::nullopt, std::nullopt, n);
  cuda::ScopedCudaDevice device(target_device);

  // Compolete inputs with defaults and broadcast constants for all required
  // parameters
  cuda::BRDFInputs params = inputs_with_defaults(
      omega_i, omega_o, P_b, P_m, std::nullopt, P_s, P_r, P_st, P_ani,
      std::nullopt, std::nullopt, std::nullopt, std::nullopt, n);

  auto *result_data =
      static_cast<float *>(cuda::cuda_allocate(params.N * 3 * sizeof(float)));
  if (result_data == nullptr) {
    throw std::runtime_error("Failed to allocate CUDA memory for output");
  }
  principled_brdf_backward_P_ani_cuda_impl(
      params.omega_i.data(), params.omega_o.data(), params.P_b.data(),
      params.P_m.data(), params.P_s.data(), params.P_r.data(),
      params.P_st.data(), params.P_ani.data(), params.n.data(), result_data,
      params.N);

  nb::capsule owner(result_data,
                    [](void *ptr) noexcept -> void { cuda::cuda_free(ptr); });
  return OutputArrayCUDA(result_data, {params.N, 3}, owner);
}

auto principled_brdf_backward_P_sh_cpu(
    const Vec3ArrayCPU &omega_i, const Vec3ArrayCPU &omega_o,
    const FlexVec3CPU &P_b = FlexVec3CPU(),
    const FlexScalarCPU &P_m = FlexScalarCPU(),
    const FlexScalarCPU &P_sht = FlexScalarCPU()) -> OutputArrayCPU {
  // Compolete inputs with defaults and broadcast constants for all required
  // parameters
  cpu::BRDFInputs params = inputs_with_defaults(
      omega_i, omega_o, P_b, P_m, std::nullopt, std::nullopt, std::nullopt,
      std::nullopt, std::nullopt, std::nullopt, P_sht, std::nullopt,
      std::nullopt, std::nullopt);

  auto *result_data = new float[params.N * 3];
  principled_brdf_backward_P_sh_cpu_impl(
      params.omega_i.data(), params.omega_o.data(), params.P_b.data(),
      params.P_m.data(), params.P_sht.data(), result_data, params.N);

  nb::capsule owner(result_data,
                    [](void *ptr) noexcept -> void { delete[] (float *)ptr; });
  return OutputArrayCPU(result_data, {params.N, 3}, owner);
}

auto principled_brdf_backward_P_sh_cuda(
    const Vec3ArrayCUDA &omega_i, const Vec3ArrayCUDA &omega_o,
    const FlexVec3CUDA &P_b = FlexVec3CUDA(),
    const FlexScalarCUDA &P_m = FlexScalarCUDA(),
    const FlexScalarCUDA &P_sht = FlexScalarCUDA()) -> OutputArrayCUDA {

  // Ensure that correct gpu is used for computation
  int target_device = cuda::get_common_cuda_device(
      omega_i, omega_o, P_b, P_m, std::nullopt, std::nullopt, std::nullopt,
      std::nullopt, std::nullopt, std::nullopt, P_sht, std::nullopt,
      std::nullopt, std::nullopt);
  cuda::ScopedCudaDevice device(target_device);

  // Compolete inputs with defaults and broadcast constants for all required
  // parameters
  cuda::BRDFInputs params = inputs_with_defaults(
      omega_i, omega_o, P_b, P_m, std::nullopt, std::nullopt, std::nullopt,
      std::nullopt, std::nullopt, std::nullopt, P_sht, std::nullopt,
      std::nullopt, std::nullopt);

  auto *result_data =
      static_cast<float *>(cuda::cuda_allocate(params.N * 3 * sizeof(float)));
  if (result_data == nullptr) {
    throw std::runtime_error("Failed to allocate CUDA memory for output");
  }
  principled_brdf_backward_P_sh_cuda_impl(
      params.omega_i.data(), params.omega_o.data(), params.P_b.data(),
      params.P_m.data(), params.P_sht.data(), result_data, params.N);

  nb::capsule owner(result_data,
                    [](void *ptr) noexcept -> void { cuda::cuda_free(ptr); });
  return OutputArrayCUDA(result_data, {params.N, 3}, owner);
}

auto principled_brdf_backward_P_sht_cpu(
    const Vec3ArrayCPU &omega_i, const Vec3ArrayCPU &omega_o,
    const FlexVec3CPU &P_b = FlexVec3CPU(),
    const FlexScalarCPU &P_m = FlexScalarCPU(),
    const FlexScalarCPU &P_sh = FlexScalarCPU()) -> OutputArrayCPU {
  // Compolete inputs with defaults and broadcast constants for all required
  // parameters
  cpu::BRDFInputs params = inputs_with_defaults(
      omega_i, omega_o, P_b, P_m, std::nullopt, std::nullopt, std::nullopt,
      std::nullopt, std::nullopt, P_sh, std::nullopt, std::nullopt,
      std::nullopt, std::nullopt);

  auto *result_data = new float[params.N * 3];
  principled_brdf_backward_P_sht_cpu_impl(
      params.omega_i.data(), params.omega_o.data(), params.P_b.data(),
      params.P_m.data(), params.P_sh.data(), result_data, params.N);

  nb::capsule owner(result_data,
                    [](void *ptr) noexcept -> void { delete[] (float *)ptr; });
  return OutputArrayCPU(result_data, {params.N, 3}, owner);
}

auto principled_brdf_backward_P_sht_cuda(
    const Vec3ArrayCUDA &omega_i, const Vec3ArrayCUDA &omega_o,
    const FlexVec3CUDA &P_b = FlexVec3CUDA(),
    const FlexScalarCUDA &P_m = FlexScalarCUDA(),
    const FlexScalarCUDA &P_sh = FlexScalarCUDA()) -> OutputArrayCUDA {

  // Ensure that correct gpu is used for computation
  int target_device = cuda::get_common_cuda_device(
      omega_i, omega_o, P_b, P_m, std::nullopt, std::nullopt, std::nullopt,
      std::nullopt, std::nullopt, P_sh, std::nullopt, std::nullopt,
      std::nullopt, std::nullopt);
  cuda::ScopedCudaDevice device(target_device);

  // Compolete inputs with defaults and broadcast constants for all required
  // parameters
  cuda::BRDFInputs params = inputs_with_defaults(
      omega_i, omega_o, P_b, P_m, std::nullopt, std::nullopt, std::nullopt,
      std::nullopt, std::nullopt, P_sh, std::nullopt, std::nullopt,
      std::nullopt, std::nullopt);

  auto *result_data =
      static_cast<float *>(cuda::cuda_allocate(params.N * 3 * sizeof(float)));
  if (result_data == nullptr) {
    throw std::runtime_error("Failed to allocate CUDA memory for output");
  }
  principled_brdf_backward_P_sht_cuda_impl(
      params.omega_i.data(), params.omega_o.data(), params.P_b.data(),
      params.P_m.data(), params.P_sh.data(), result_data, params.N);

  nb::capsule owner(result_data,
                    [](void *ptr) noexcept -> void { cuda::cuda_free(ptr); });
  return OutputArrayCUDA(result_data, {params.N, 3}, owner);
}

auto principled_brdf_backward_P_c_cpu(
    const Vec3ArrayCPU &omega_i, const Vec3ArrayCPU &omega_o,
    const FlexScalarCPU &P_cg = FlexScalarCPU(),
    const FlexVec3CPU &n = FlexVec3CPU()) -> OutputArrayCPU {
  // Compolete inputs with defaults and broadcast constants for all required
  // parameters
  cpu::BRDFInputs params = inputs_with_defaults(
      omega_i, omega_o, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
      std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
      std::nullopt, P_cg, n);

  auto *result_data = new float[params.N * 3];
  principled_brdf_backward_P_c_cpu_impl(
      params.omega_i.data(), params.omega_o.data(), params.P_cg.data(),
      params.n.data(), result_data, params.N);

  nb::capsule owner(result_data,
                    [](void *ptr) noexcept -> void { delete[] (float *)ptr; });
  return OutputArrayCPU(result_data, {params.N, 3}, owner);
}

auto principled_brdf_backward_P_c_cuda(
    const Vec3ArrayCUDA &omega_i, const Vec3ArrayCUDA &omega_o,
    const FlexScalarCUDA &P_cg = FlexScalarCUDA(),
    const FlexVec3CUDA &n = FlexVec3CUDA()) -> OutputArrayCUDA {

  // Ensure that correct gpu is used for computation
  int target_device = cuda::get_common_cuda_device(
      omega_i, omega_o, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
      std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
      std::nullopt, P_cg, n);
  cuda::ScopedCudaDevice device(target_device);

  // Compolete inputs with defaults and broadcast constants for all required
  // parameters
  cuda::BRDFInputs params = inputs_with_defaults(
      omega_i, omega_o, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
      std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
      std::nullopt, P_cg, n);

  auto *result_data =
      static_cast<float *>(cuda::cuda_allocate(params.N * 3 * sizeof(float)));
  if (result_data == nullptr) {
    throw std::runtime_error("Failed to allocate CUDA memory for output");
  }
  principled_brdf_backward_P_c_cuda_impl(
      params.omega_i.data(), params.omega_o.data(), params.P_cg.data(),
      params.n.data(), result_data, params.N);

  nb::capsule owner(result_data,
                    [](void *ptr) noexcept -> void { cuda::cuda_free(ptr); });
  return OutputArrayCUDA(result_data, {params.N, 3}, owner);
}

NB_MODULE(principled_brdf_functions, module) {
  module.doc() = "Raw Principled BRDF functions with containing functions for "
                 "forward pass and partial derivatives wrt. all parameters and "
                 "normals as fused kernel.";

  // Two separate overloads - nanobind will automatically dispatch based on
  // device type
  module.def(
      "principled_brdf_forward", &principled_brdf_forward_cpu,
      "CPU implementation of the Principled BRDF forward pass.\n\n"
      "This is used when all arguments are on cpu.\n"
      "Args:\n"
      "    omega_i (L): Direction towards incoming light [N, 3]\n"
      "    omega_o (V): Direction towards viewer [N, 3]\n"
      "    basecolor (P_b): Base color [N, 3] or [1, 3] (default: [0.8, 0.8, "
      "0.8])\n"
      "    metallic (P_m): Metallic [N] or [1] (default: 0.0)\n"
      "    subsurface (P_ss): Subsurface [N] or [1] (default: 0.0)\n"
      "    specular (P_s): Specular [N] or [1] (default: 0.5)\n"
      "    roughness (P_r): Roughness [N] or [1] (default: 0.5)\n"
      "    specular_tint (P_st): Specular tint [N] or [1] (default: 0.0)\n"
      "    anisotropy (P_ani): Anisotropic [N] or [1] (default: 0.0)\n"
      "    sheen (P_sh): Sheen [N] or [1] (default: 0.0)\n"
      "    sheen_tint (P_sht): Sheen tint [N] or [1] (default: 0.5)\n"
      "    clearcoat (P_c): Clearcoat [N] or [1] (default: 0.0)\n"
      "    clearcoat_gloss (P_cg): Clearcoat gloss [N] or [1] (default: 1.0)\n"
      "    normal (n): Surface normal [N, 3] or [1, 3] (default: [0, 0, 1])\n\n"
      "Returns:\n"
      "    BRDF value in rgb [N, 3]",
      "omega_i"_a, "omega_o"_a, nb::kw_only(), "basecolor"_a = FlexVec3CPU(),
      "metallic"_a = FlexScalarCPU(), "subsurface"_a = FlexScalarCPU(),
      "specular"_a = FlexScalarCPU(), "roughness"_a = FlexScalarCPU(),
      "specular_tint"_a = FlexScalarCPU(), "anisotropy"_a = FlexScalarCPU(),
      "sheen"_a = FlexScalarCPU(), "sheen_tint"_a = FlexScalarCPU(),
      "clearcoat"_a = FlexScalarCPU(), "clearcoat_gloss"_a = FlexScalarCPU(),
      "normal"_a = FlexVec3CPU());

  module.def(
      "principled_brdf_forward", &principled_brdf_forward_cuda,
      "GPU implementation of the Principled BRDF forward pass.\n\n"
      "This is used when all arguments are on gpu.\n"
      "Args:\n"
      "    omega_i (L): Incoming light direction [N, 3]\n"
      "    omega_o (V): Outgoing view direction [N, 3]\n"
      "    basecolor (P_b): Base color [N, 3] or [1, 3] (default: [0.8, 0.8, "
      "0.8])\n"
      "    metallic (P_m): Metallic [N] or [1] (default: 0.0)\n"
      "    subsurface (P_ss): Subsurface [N] or [1] (default: 0.0)\n"
      "    specular (P_s): Specular [N] or [1] (default: 0.5)\n"
      "    roughness (P_r): Roughness [N] or [1] (default: 0.5)\n"
      "    specular_tint (P_st): Specular tint [N] or [1] (default: 0.0)\n"
      "    anisotropy (P_ani): Anisotropic [N] or [1] (default: 0.0)\n"
      "    sheen (P_sh): Sheen [N] or [1] (default: 0.0)\n"
      "    sheen_tint (P_sht): Sheen tint [N] or [1] (default: 0.5)\n"
      "    clearcoat (P_c): Clearcoat [N] or [1] (default: 0.0)\n"
      "    clearcoat_gloss (P_cg): Clearcoat gloss [N] or [1] (default: 1.0)\n"
      "    normal (n): Surface normal [N, 3] or [1, 3] (default: [0, 0, 1])\n\n"
      "Returns:\n"
      "    BRDF value in rgb [N, 3]",
      "omega_i"_a, "omega_o"_a, nb::kw_only(), "basecolor"_a = FlexVec3CUDA(),
      "metallic"_a = FlexScalarCUDA(), "subsurface"_a = FlexScalarCUDA(),
      "specular"_a = FlexScalarCUDA(), "roughness"_a = FlexScalarCUDA(),
      "specular_tint"_a = FlexScalarCUDA(), "anisotropy"_a = FlexScalarCUDA(),
      "sheen"_a = FlexScalarCUDA(), "sheen_tint"_a = FlexScalarCUDA(),
      "clearcoat"_a = FlexScalarCUDA(), "clearcoat_gloss"_a = FlexScalarCUDA(),
      "normal"_a = FlexVec3CUDA());

  module.def(
      "principled_brdf_backward_basecolor", &principled_brdf_backward_P_b_cpu,
      "CPU implementation of the partial derivative of the Principled BRDF "
      "w.r.t. the basecolor parameter P_b.\n\n"
      "This implementation is used when all arguments are on gpu.\n"
      "Args:\n"
      "    omega_i: Incoming light direction [N, 3]\n"
      "    omega_o: Outgoing view direction [N, 3]\n"
      "    basecolor: Base color [N, 3] or [1, 3] (default: [0.8, 0.8, 0.8])\n"
      "    metallic: Metallic [N] or [1] (default: 0.0)\n"
      "    subsurface: Subsurface [N] or [1] (default: 0.0)\n"
      "    specular: Specular [N] or [1] (default: 0.5)\n"
      "    roughness: Roughness [N] or [1] (default: 0.5)\n"
      "    specular_tint: Specular tint [N] or [1] (default: 0.0)\n"
      "    anisotropy: Anisotropic [N] or [1] (default: 0.0)\n"
      "    sheen: Sheen [N] or [1] (default: 0.0)\n"
      "    sheen_tint: Sheen tint [N] or [1] (default: 0.5)\n"
      "    normal: Surface normal [N, 3] or [1, 3] (default: [0, 0, 1])\n\n"
      "Returns:\n"
      "    Partial derivative dBRDF/dP_b value in rgb [N, 3]\n\n"
      "Note:\n"
      "    clearcoat and clearcoat_gloss are NOT needed!\n",
      "omega_i"_a, "omega_o"_a, nb::kw_only(), "basecolor"_a = FlexVec3CPU(),
      "metallic"_a = FlexScalarCPU(), "subsurface"_a = FlexScalarCPU(),
      "specular"_a = FlexScalarCPU(), "roughness"_a = FlexScalarCPU(),
      "specular_tint"_a = FlexScalarCPU(), "anisotropy"_a = FlexScalarCPU(),
      "sheen"_a = FlexScalarCPU(), "sheen_tint"_a = FlexScalarCPU(),
      "normal"_a = FlexVec3CPU());

  module.def(
      "principled_brdf_backward_basecolor", &principled_brdf_backward_P_b_cuda,
      "GPU implementation of the partial derivative of the Principled BRDF "
      "w.r.t. the basecolor parameter P_b.\n\n"
      "This implementation is used when all arguments are on gpu.\n"
      "Args:\n"
      "    omega_i: Incoming light direction [N, 3]\n"
      "    omega_o: Outgoing view direction [N, 3]\n"
      "    basecolor: Base color [N, 3] or [1, 3] (default: [0.8, 0.8, 0.8])\n"
      "    metallic: Metallic [N] or [1] (default: 0.0)\n"
      "    subsurface: Subsurface [N] or [1] (default: 0.0)\n"
      "    specular: Specular [N] or [1] (default: 0.5)\n"
      "    roughness: Roughness [N] or [1] (default: 0.5)\n"
      "    specular_tint: Specular tint [N] or [1] (default: 0.0)\n"
      "    anisotropy: Anisotropic [N] or [1] (default: 0.0)\n"
      "    sheen: Sheen [N] or [1] (default: 0.0)\n"
      "    sheen_tint: Sheen tint [N] or [1] (default: 0.5)\n"
      "    normal: Surface normal [N, 3] or [1, 3] (default: [0, 0, 1])\n\n"
      "Returns:\n"
      "    Partial derivative dBRDF/dP_b value in rgb [N, 3]\n\n"
      "Note:\n"
      "    clearcoat and clearcoat_gloss are NOT needed!\n",
      "omega_i"_a, "omega_o"_a, nb::kw_only(), "basecolor"_a = FlexVec3CUDA(),
      "metallic"_a = FlexScalarCUDA(), "subsurface"_a = FlexScalarCUDA(),
      "specular"_a = FlexScalarCUDA(), "roughness"_a = FlexScalarCUDA(),
      "specular_tint"_a = FlexScalarCUDA(), "anisotropy"_a = FlexScalarCUDA(),
      "sheen"_a = FlexScalarCUDA(), "sheen_tint"_a = FlexScalarCUDA(),
      "normal"_a = FlexVec3CUDA());

  module.def(
      "principled_brdf_backward_metallic", &principled_brdf_backward_P_m_cpu,
      "CPU implementation of the partial derivative of the Principled BRDF "
      "w.r.t. the metallic parameter P_m.\n\n"
      "This implementation is used when all arguments are on cpu.\n"
      "Args:\n"
      "    omega_i (L): Direction towards incoming light [N, 3]\n"
      "    omega_o (V): Direction towards viewer [N, 3]\n"
      "    basecolor (P_b): Base color [N, 3] or [1, 3] (default: [0.8, 0.8, "
      "0.8])\n"
      "    subsurface (P_ss): Subsurface [N] or [1] (default: 0.0)\n"
      "    specular (P_s): Specular [N] or [1] (default: 0.5)\n"
      "    roughness (P_r): Roughness [N] or [1] (default: 0.5)\n"
      "    specular_tint (P_st): Specular tint [N] or [1] (default: 0.0)\n"
      "    anisotropy (P_ani): Anisotropic [N] or [1] (default: 0.0)\n"
      "    sheen (P_sh): Sheen [N] or [1] (default: 0.0)\n"
      "    sheen_tint (P_sht): Sheen tint [N] or [1] (default: 0.5)\n"
      "    normal (n): Surface normal [N, 3] or [1, 3] (default: [0, 0, 1])\n\n"
      "Returns:\n"
      "    Partial derivative dBRDF/dP_b value in rgb [N, 3]\n\n"
      "Note:\n"
      "    metallic, clearcoat and clearcoat_gloss are NOT needed!\n",
      "omega_i"_a, "omega_o"_a, nb::kw_only(), "basecolor"_a = FlexVec3CPU(),
      "subsurface"_a = FlexScalarCPU(), "specular"_a = FlexScalarCPU(),
      "roughness"_a = FlexScalarCPU(), "specular_tint"_a = FlexScalarCPU(),
      "anisotropy"_a = FlexScalarCPU(), "sheen"_a = FlexScalarCPU(),
      "sheen_tint"_a = FlexScalarCPU(), "normal"_a = FlexVec3CPU());

  module.def(
      "principled_brdf_backward_metallic", &principled_brdf_backward_P_m_cuda,
      "GPU implementation of the partial derivative of the Principled BRDF "
      "w.r.t. the metallic parameter P_m.\n\n"
      "This implementation is used when all arguments are on gpu.\n"
      "Args:\n"
      "    omega_i (L): Direction towards incoming light [N, 3]\n"
      "    omega_o (V): Direction towards viewer [N, 3]\n"
      "    basecolor (P_b): Base color [N, 3] or [1, 3] (default: [0.8, 0.8, "
      "0.8])\n"
      "    subsurface (P_ss): Subsurface [N] or [1] (default: 0.0)\n"
      "    specular (P_s): Specular [N] or [1] (default: 0.5)\n"
      "    roughness (P_r): Roughness [N] or [1] (default: 0.5)\n"
      "    specular_tint (P_st): Specular tint [N] or [1] (default: 0.0)\n"
      "    anisotropy (P_ani): Anisotropic [N] or [1] (default: 0.0)\n"
      "    sheen (P_sh): Sheen [N] or [1] (default: 0.0)\n"
      "    sheen_tint (P_sht): Sheen tint [N] or [1] (default: 0.5)\n"
      "    normal (n): Surface normal [N, 3] or [1, 3] (default: [0, 0, 1])\n\n"
      "Returns:\n"
      "    Partial derivative dBRDF/dP_b value in rgb [N, 3]\n\n"
      "Note:\n"
      "    metallic, clearcoat and clearcoat_gloss are NOT needed!\n",
      "omega_i"_a, "omega_o"_a, nb::kw_only(), "basecolor"_a = FlexVec3CUDA(),
      "subsurface"_a = FlexScalarCUDA(), "specular"_a = FlexScalarCUDA(),
      "roughness"_a = FlexScalarCUDA(), "specular_tint"_a = FlexScalarCUDA(),
      "anisotropy"_a = FlexScalarCUDA(), "sheen"_a = FlexScalarCUDA(),
      "sheen_tint"_a = FlexScalarCUDA(), "normal"_a = FlexVec3CUDA());

  module.def(
      "principled_brdf_backward_subsurface", &principled_brdf_backward_P_ss_cpu,
      "CPU implementation of the partial derivative of the Principled BRDF "
      "w.r.t. the subsurface parameter P_ss.\n\n"
      "This implementation is used when all arguments are on cpu.\n"
      "Args:\n"
      "    omega_i (L): Direction towards incoming light [N, 3]\n"
      "    omega_o (V): Direction towards viewer [N, 3]\n"
      "    basecolor (P_b): Base color [N, 3] or [1, 3] (default: [0.8, 0.8, "
      "    metallic: Metallic [N] or [1] (default: 0.0)\n"
      "0.8])\n"
      "    roughness (P_r): Roughness [N] or [1] (default: 0.5)\n"
      "    normal (n): Surface normal [N, 3] or [1, 3] (default: [0, 0, 1])\n\n"
      "Returns:\n"
      "    Partial derivative dBRDF/dP_ss value in rgb [N, 3]\n\n"
      "Note:\n"
      "    subsurface, specular, specular_tint, anisotropy, sheen, sheen_tint, "
      "clearcoat and clearcoat_gloss are NOT needed!\n",
      "omega_i"_a, "omega_o"_a, nb::kw_only(), "basecolor"_a = FlexVec3CPU(),
      "metallic"_a = FlexScalarCPU(), "roughness"_a = FlexScalarCPU(),
      "normal"_a = FlexVec3CPU());

  module.def(
      "principled_brdf_backward_metallic", &principled_brdf_backward_P_ss_cuda,
      "GPU implementation of the partial derivative of the Principled BRDF "
      "w.r.t. the subsurface parameter P_ss.\n\n"
      "This implementation is used when all arguments are on gpu.\n"
      "Args:\n"
      "    omega_i (L): Direction towards incoming light [N, 3]\n"
      "    omega_o (V): Direction towards viewer [N, 3]\n"
      "    basecolor (P_b): Base color [N, 3] or [1, 3] (default: [0.8, 0.8, "
      "0.8])\n"
      "    metallic: Metallic [N] or [1] (default: 0.0)\n"
      "    roughness (P_r): Roughness [N] or [1] (default: 0.5)\n"
      "    normal (n): Surface normal [N, 3] or [1, 3] (default: [0, 0, 1])\n\n"
      "Returns:\n"
      "    Partial derivative dBRDF/dP_ss value in rgb [N, 3]\n\n"
      "Note:\n"
      "    subsurface, specular, specular_tint, anisotropy, sheen, sheen_tint, "
      "clearcoat and clearcoat_gloss are NOT needed!\n",
      "omega_i"_a, "omega_o"_a, nb::kw_only(), "basecolor"_a = FlexVec3CUDA(),
      "metallic"_a = FlexScalarCUDA(), "roughness"_a = FlexScalarCUDA(),
      "normal"_a = FlexVec3CUDA());

  module.def(
      "principled_brdf_backward_specular", &principled_brdf_backward_P_s_cpu,
      "CPU implementation of the partial derivative of the Principled BRDF "
      "w.r.t. the specular parameter P_s.\n\n"
      "This implementation is used when all arguments are on cpu.\n"
      "Args:\n"
      "    omega_i (L): Direction towards incoming light [N, 3]\n"
      "    omega_o (V): Direction towards viewer [N, 3]\n"
      "    basecolor (P_b): Base color [N, 3] or [1, 3] (default: [0.8, 0.8, "
      "0.8])\n"
      "    metallic (P_m): Metallic [N] or [1] (default: 0.0)\n"
      "    subsurface (P_ss): Subsurface [N] or [1] (default: 0.0)\n"
      "    specular (P_s): Specular [N] or [1] (default: 0.5)\n"
      "    roughness (P_r): Roughness [N] or [1] (default: 0.5)\n"
      "    specular_tint (P_st): Specular tint [N] or [1] (default: 0.0)\n"
      "    anisotropy (P_ani): Anisotropic [N] or [1] (default: 0.0)\n"
      "    sheen (P_sh): Sheen [N] or [1] (default: 0.0)\n"
      "    sheen_tint (P_sht): Sheen tint [N] or [1] (default: 0.5)\n"
      "    clearcoat (P_c): Clearcoat [N] or [1] (default: 0.0)\n"
      "    clearcoat_gloss (P_cg): Clearcoat gloss [N] or [1] (default: 1.0)\n"
      "    normal (n): Surface normal [N, 3] or [1, 3] (default: [0, 0, 1])\n\n"
      "Returns:\n"
      "    Partial derivative dBRDF/dP_s value in rgb [N, 3]\n\n"
      "Note:\n"
      "    subsurface, specular, roughness, anisotropy, sheen, sheen_tint, "
      "clearcoat and clearcoat_gloss are NOT needed!\n",
      "omega_i"_a, "omega_o"_a, nb::kw_only(), "basecolor"_a = FlexVec3CPU(),
      "metallic"_a = FlexScalarCPU(), "specular_tint"_a = FlexScalarCPU(),
      "normal"_a = FlexVec3CPU());

  module.def(
      "principled_brdf_forward", &principled_brdf_backward_P_s_cuda,
      "GPU implementation of the partial derivative of the Principled BRDF "
      "w.r.t. the subsurface parameter P_ss.\n\n"
      "This implementation is used when all arguments are on gpu.\n"
      "Args:\n"
      "    omega_i: Incoming light direction [N, 3]\n"
      "    omega_o: Outgoing view direction [N, 3]\n"
      "    basecolor: Base color [N, 3] or [1, 3] (default: [0.8, 0.8, 0.8])\n"
      "    metallic: Metallic [N] or [1] (default: 0.0)\n"
      "    specular_tint: Specular tint [N] or [1] (default: 0.0)\n"
      "    normal: Surface normal [N, 3] or [1, 3] (default: [0, 0, 1])\n\n"
      "Returns:\n"
      "    Partial derivative dBRDF/dP_s value in rgb [N, 3]\n\n"
      "Note:\n"
      "    subsurface, specular, roughness, anisotropy, sheen, sheen_tint, "
      "clearcoat and clearcoat_gloss are NOT needed!\n",
      "omega_i"_a, "omega_o"_a, nb::kw_only(), "basecolor"_a = FlexVec3CUDA(),
      "metallic"_a = FlexScalarCUDA(), "specular_tint"_a = FlexScalarCUDA(),
      "normal"_a = FlexVec3CUDA());

  module.def(
      "principled_brdf_backward_roughness", &principled_brdf_backward_P_r_cpu,
      "CPU implementation of the partial derivative of the Principled BRDF "
      "w.r.t. the roughness parameter P_r.\n\n"
      "This implementation is used when all arguments are on cpu.\n"
      "Args:\n"
      "    omega_i (L): Direction towards incoming light [N, 3]\n"
      "    omega_o (V): Direction towards viewer [N, 3]\n"
      "    basecolor (P_b): Base color [N, 3] or [1, 3] (default: [0.8, 0.8, "
      "0.8])\n"
      "    metallic (P_m): Metallic [N] or [1] (default: 0.0)\n"
      "    subsurface (P_ss): Subsurface [N] or [1] (default: 0.0)\n"
      "    specular (P_s): Specular [N] or [1] (default: 0.5)\n"
      "    roughness (P_r): Roughness [N] or [1] (default: 0.5)\n"
      "    specular_tint (P_st): Specular tint [N] or [1] (default: 0.0)\n"
      "    anisotropy (P_ani): Anisotropic [N] or [1] (default: 0.0)\n"
      "    normal (n): Surface normal [N, 3] or [1, 3] (default: [0, 0, 1])\n\n"
      "Returns:\n"
      "    Partial derivative dBRDF/dP_r value in rgb [N, 3]\n\n"
      "Note:\n"
      "sheen, sheen_tint, clearcoat and clearcoat_gloss are not needed.\n",
      "omega_i"_a, "omega_o"_a, nb::kw_only(), "basecolor"_a = FlexVec3CPU(),
      "metallic"_a = FlexScalarCPU(), "subsurface"_a = FlexScalarCPU(),
      "specular"_a = FlexScalarCPU(), "roughness"_a = FlexScalarCPU(),
      "specular_tint"_a = FlexScalarCPU(), "anisotropy"_a = FlexScalarCPU(),
      "normal"_a = FlexVec3CPU());

  module.def(
      "principled_brdf_backward_roughness", &principled_brdf_backward_P_r_cuda,
      "CPU implementation of the partial derivative of the Principled BRDF "
      "w.r.t. the roughness parameter P_r.\n\n"
      "This implementation is used when all arguments are on cpu.\n"
      "Args:\n"
      "    omega_i: Incoming light direction [N, 3]\n"
      "    omega_o: Outgoing view direction [N, 3]\n"
      "    basecolor: Base color [N, 3] or [1, 3] (default: [0.8, 0.8, 0.8])\n"
      "    metallic (P_m): Metallic [N] or [1] (default: 0.0)\n"
      "    subsurface (P_ss): Subsurface [N] or [1] (default: 0.0)\n"
      "    specular (P_s): Specular [N] or [1] (default: 0.5)\n"
      "    roughness (P_r): Roughness [N] or [1] (default: 0.5)\n"
      "    specular_tint (P_st): Specular tint [N] or [1] (default: 0.0)\n"
      "    anisotropy (P_ani): Anisotropic [N] or [1] (default: 0.0)\n"
      "    normal (n): Surface normal [N, 3] or [1, 3] (default: [0, 0, 1])\n\n"
      "Returns:\n"
      "    Partial derivative dBRDF/dP_r value in rgb [N, 3]\n\n"
      "Note:\n"
      "sheen, sheen_tint, clearcoat and clearcoat_gloss are not needed.\n",
      "omega_i"_a, "omega_o"_a, nb::kw_only(), "basecolor"_a = FlexVec3CUDA(),
      "metallic"_a = FlexScalarCUDA(), "subsurface"_a = FlexScalarCUDA(),
      "specular"_a = FlexScalarCUDA(), "roughness"_a = FlexScalarCUDA(),
      "specular_tint"_a = FlexScalarCUDA(), "anisotropy"_a = FlexScalarCUDA(),
      "normal"_a = FlexVec3CUDA());

  module.def(
      "principled_brdf_backward_specular_tint",
      &principled_brdf_backward_P_st_cpu,
      "CPU implementation of the partial derivative of the Principled BRDF "
      "w.r.t. the specular tint parameter P_st.\n\n"
      "This implementation is used when all arguments are on cpu.\n"
      "Args:\n"
      "    omega_i (L): Direction towards incoming light [N, 3]\n"
      "    omega_o (V): Direction towards viewer [N, 3]\n"
      "    basecolor (P_b): Base color [N, 3] or [1, 3] (default: [0.8, 0.8, "
      "0.8])\n"
      "    metallic (P_m): Metallic [N] or [1] (default: 0.0)\n"
      "    specular (P_s): Specular [N] or [1] (default: 0.5)\n"
      "    roughness (P_r): Roughness [N] or [1] (default: 0.5)\n"
      "    anisotropy (P_ani): Anisotropic [N] or [1] (default: 0.0)\n"
      "    normal (n): Surface normal [N, 3] or [1, 3] (default: [0, 0, 1])\n\n"
      "Returns:\n"
      "    Partial derivative dBRDF/dP_st value in rgb [N, 3]\n\n"
      "Note:\n"
      "    subsurface, specular_tint, sheen, sheen_tint, clearcoat and "
      "clearcoat_gloss are not needed.\n",
      "omega_i"_a, "omega_o"_a, nb::kw_only(), "basecolor"_a = FlexVec3CPU(),
      "metallic"_a = FlexScalarCPU(), "specular"_a = FlexScalarCPU(),
      "roughness"_a = FlexScalarCPU(), "anisotropy"_a = FlexScalarCPU(),
      "normal"_a = FlexVec3CPU());

  module.def(
      "principled_brdf_backward_specular_tint",
      &principled_brdf_backward_P_st_cuda,
      "GPU implementation of the partial derivative of the Principled BRDF "
      "w.r.t. the specular tint parameter P_st.\n\n"
      "This implementation is used when all arguments are on gpu.\n"
      "Args:\n"
      "    omega_i (L): Direction towards incoming light [N, 3]\n"
      "    omega_o (V): Direction towards viewer [N, 3]\n"
      "    basecolor (P_b): Base color [N, 3] or [1, 3] (default: [0.8, 0.8, "
      "0.8])\n"
      "    metallic (P_m): Metallic [N] or [1] (default: 0.0)\n"
      "    specular (P_s): Specular [N] or [1] (default: 0.5)\n"
      "    roughness (P_r): Roughness [N] or [1] (default: 0.5)\n"
      "    anisotropy (P_ani): Anisotropic [N] or [1] (default: 0.0)\n"
      "    normal (n): Surface normal [N, 3] or [1, 3] (default: [0, 0, 1])\n\n"
      "Returns:\n"
      "    Partial derivative dBRDF/dP_st value in rgb [N, 3]\n\n"
      "Note:\n"
      "    subsurface, specular_tint, sheen, sheen_tint, clearcoat and "
      "clearcoat_gloss are not needed.\n",
      "omega_i"_a, "omega_o"_a, nb::kw_only(), "basecolor"_a = FlexVec3CUDA(),
      "metallic"_a = FlexScalarCUDA(), "specular"_a = FlexScalarCUDA(),
      "roughness"_a = FlexScalarCUDA(), "anisotropy"_a = FlexScalarCUDA(),
      "normal"_a = FlexVec3CUDA());

  module.def(
      "principled_brdf_backward_anisotropy",
      &principled_brdf_backward_P_ani_cpu,
      "CPU implementation of the partial derivative of the Principled BRDF "
      "w.r.t. the anisotropy parameter P_ani.\n\n"
      "This implementation is used when all arguments are on cpu.\n"
      "Args:\n"
      "    omega_i (L): Direction towards incoming light [N, 3]\n"
      "    omega_o (V): Direction towards viewer [N, 3]\n"
      "    basecolor (P_b): Base color [N, 3] or [1, 3] (default: [0.8, 0.8, "
      "0.8])\n"
      "    metallic (P_m): Metallic [N] or [1] (default: 0.0)\n"
      "    specular (P_s): Specular [N] or [1] (default: 0.5)\n"
      "    roughness (P_r): Roughness [N] or [1] (default: 0.5)\n"
      "    specular_tint (P_st): Specular tint [N] or [1] (default: 0.0)\n"
      "    anisotropy (P_ani): Anisotropic [N] or [1] (default: 0.0)\n"
      "    normal (n): Surface normal [N, 3] or [1, 3] (default: [0, 0, 1])\n\n"
      "Returns:\n"
      "    Partial derivative dBRDF/dP_ani value in rgb [N, 3]\n\n"
      "Note:\n"
      "    subsurface, sheen, sheen_tint, clearcoat, clearcoat_gloss are not "
      "needed.",
      "omega_i"_a, "omega_o"_a, nb::kw_only(), "basecolor"_a = FlexVec3CPU(),
      "metallic"_a = FlexScalarCPU(), "specular"_a = FlexScalarCPU(),
      "roughness"_a = FlexScalarCPU(), "specular_tint"_a = FlexScalarCPU(),
      "anisotropy"_a = FlexScalarCPU(), "normal"_a = FlexVec3CPU());

  module.def(
      "principled_brdf_backward_anisotropy",
      &principled_brdf_backward_P_ani_cuda,
      "GPU implementation of the partial derivative of the Principled BRDF "
      "w.r.t. the anisotropy parameter P_ani.\n\n"
      "This implementation is used when all arguments are on gpu.\n"
      "Args:\n"
      "    omega_i: Incoming light direction [N, 3]\n"
      "    omega_o: Outgoing view direction [N, 3]\n"
      "    basecolor: Base color [N, 3] or [1, 3] (default: [0.8, 0.8, 0.8])\n"
      "    metallic (P_m): Metallic [N] or [1] (default: 0.0)\n"
      "    specular (P_s): Specular [N] or [1] (default: 0.5)\n"
      "    roughness (P_r): Roughness [N] or [1] (default: 0.5)\n"
      "    specular_tint (P_st): Specular tint [N] or [1] (default: 0.0)\n"
      "    anisotropy (P_ani): Anisotropic [N] or [1] (default: 0.0)\n"
      "    normal (n): Surface normal [N, 3] or [1, 3] (default: [0, 0, 1])\n\n"
      "Returns:\n"
      "    Partial derivative dBRDF/dP_ani value in rgb [N, 3]\n\n"
      "Note:\n"
      "    subsurface, sheen, sheen_tint, clearcoat, clearcoat_gloss are not "
      "needed.",
      "omega_i"_a, "omega_o"_a, nb::kw_only(), "basecolor"_a = FlexVec3CUDA(),
      "metallic"_a = FlexScalarCUDA(), "specular"_a = FlexScalarCUDA(),
      "roughness"_a = FlexScalarCUDA(), "specular_tint"_a = FlexScalarCUDA(),
      "anisotropy"_a = FlexScalarCUDA(), "normal"_a = FlexVec3CUDA());

  module.def(
      "principled_brdf_backward_sheen", &principled_brdf_backward_P_sh_cpu,
      "CPU implementation of the partial derivative of the Principled BRDF "
      "w.r.t. the sheen parameter P_sh.\n\n"
      "This implementation is used when all arguments are on cpu.\n"
      "Args:\n"
      "    omega_i (L): Direction towards incoming light [N, 3]\n"
      "    omega_o (V): Direction towards viewer [N, 3]\n"
      "    basecolor (P_b): Base color [N, 3] or [1, 3] (default: [0.8, 0.8, "
      "0.8])\n"
      "    metallic (P_m): Metallic [N] or [1] (default: 0.0)\n"
      "    sheen_tint (P_sht): Sheen tint [N] or [1] (default: 0.5)\n\n"
      "Returns:\n"
      "    Partial derivative dBRDF/dP_sh value in rgb [N, 3]\n\n"
      "omega_i"_a,
      "omega_o"_a, nb::kw_only(), "basecolor"_a = FlexVec3CPU(),
      "metallic"_a = FlexScalarCPU(), "sheen_tint"_a = FlexScalarCPU());

  module.def(
      "principled_brdf_backward_sheen", &principled_brdf_backward_P_sh_cuda,
      "GPU implementation of the partial derivative of the Principled BRDF "
      "w.r.t. the sheen parameter P_sh.\n\n"
      "This implementation is used when all arguments are on gpu.\n"
      "Args:\n"
      "    omega_i: Incoming light direction [N, 3]\n"
      "    omega_o: Outgoing view direction [N, 3]\n"
      "    basecolor: Base color [N, 3] or [1, 3] (default: [0.8, 0.8, 0.8])\n"
      "    metallic: Metallic [N] or [1] (default: 0.0)\n"
      "    sheen_tint: Sheen tint [N] or [1] (default: 0.5)\n"
      "Returns:\n"
      "    Partial derivative dBRDF/dP_sh value in rgb [N, 3]\n\n"
      "omega_i"_a,
      "omega_o"_a, nb::kw_only(), "basecolor"_a = FlexVec3CUDA(),
      "metallic"_a = FlexScalarCUDA(), "sheen_tint"_a = FlexScalarCUDA());

  module.def(
      "principled_brdf_backward_sheen_tint",
      &principled_brdf_backward_P_sht_cpu,
      "CPU implementation of the partial derivative of the Principled BRDF "
      "w.r.t. the sheen_tint parameter P_sht.\n\n"
      "This implementation is used when all arguments are on cpu.\n"
      "Args:\n"
      "    omega_i (L): Direction towards incoming light [N, 3]\n"
      "    omega_o (V): Direction towards viewer [N, 3]\n"
      "    basecolor (P_b): Base color [N, 3] or [1, 3] (default: [0.8, 0.8, "
      "0.8])\n"
      "    metallic (P_m): Metallic [N] or [1] (default: 0.0)\n"
      "    sheen (P_sh): Sheen [N] or [1] (default: 0.0)\n\n"
      "Returns:\n"
      "    Partial derivative dBRDF/dP_sht value in rgb [N, 3]\n\n"
      "omega_i"_a,
      "omega_o"_a, nb::kw_only(), "basecolor"_a = FlexVec3CPU(),
      "metallic"_a = FlexScalarCPU(), "sheen"_a = FlexScalarCPU());

  module.def(
      "principled_brdf_backward_sheen", &principled_brdf_backward_P_sht_cuda,
      "GPU implementation of the partial derivative of the Principled BRDF "
      "w.r.t. the sheen_tint parameter P_sht.\n\n"
      "This implementation is used when all arguments are on gpu.\n"
      "Args:\n"
      "    omega_i: Incoming light direction [N, 3]\n"
      "    omega_o: Outgoing view direction [N, 3]\n"
      "    basecolor: Base color [N, 3] or [1, 3] (default: [0.8, 0.8, 0.8])\n"
      "    metallic: Metallic [N] or [1] (default: 0.0)\n"
      "    sheen: Sheen [N] or [1] (default: 0.0)\n"
      "Returns:\n"
      "    Partial derivative dBRDF/dP_sht value in rgb [N, 3]\n\n"
      "omega_i"_a,
      "omega_o"_a, nb::kw_only(), "basecolor"_a = FlexVec3CUDA(),
      "metallic"_a = FlexScalarCUDA(), "sheen"_a = FlexScalarCUDA());

  module.def(
      "principled_brdf_backward_clearcoat", &principled_brdf_backward_P_c_cpu,
      "CPU implementation of the Principled BRDF forward pass.\n\n"
      "This is used when all arguments are on cpu.\n"
      "Args:\n"
      "    omega_i (L): Direction towards incoming light [N, 3]\n"
      "    omega_o (V): Direction towards viewer [N, 3]\n"
      "    clearcoat_gloss (P_cg): Clearcoat gloss [N] or [1] (default: 1.0)\n"
      "    normal (n): Surface normal [N, 3] or [1, 3] (default: [0, 0, 1])\n\n"
      "Returns:\n"
      "    Partial derivative dBRDF/dP_c value in rgb [N, 3]\n\n"
      "omega_i"_a, "omega_o"_a, nb::kw_only(),
      "clearcoat_gloss"_a = FlexScalarCPU(), "normal"_a = FlexVec3CPU());

  module.def(
      "principled_brdf_backward_clearcoat", &principled_brdf_backward_P_c_cuda,
      "GPU implementation of the partial derivative of the Principled BRDF "
      "w.r.t. the clearcoat parameter P_c.\n\n"
      "This implementation is used when all arguments are on gpu.\n"
      "Args:\n"
      "    omega_i (L): Incoming light direction [N, 3]\n"
      "    omega_o (V): Outgoing view direction [N, 3]\n"
      "    clearcoat_gloss (P_cg): Clearcoat gloss [N] or [1] (default: 1.0)\n"
      "    normal (n): Surface normal [N, 3] or [1, 3] (default: [0, 0, 1])\n\n"
      "Returns:\n"
      "    Partial derivative dBRDF/dP_c value in rgb [N, 3]\n\n"
      "omega_i"_a, "omega_o"_a, nb::kw_only(),
      "clearcoat_gloss"_a = FlexScalarCUDA(), "normal"_a = FlexVec3CUDA());
}
