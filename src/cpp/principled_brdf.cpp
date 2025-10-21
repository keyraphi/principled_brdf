#include <cstddef>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <stdexcept>
#include <optional>

#include "common/utils.h"
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

auto inputs_with_defaults(const BRDFInputRequirements &requirements,
                          const Vec3ArrayCUDA &omega_i,
                          const Vec3ArrayCUDA &omega_o,
                          const FlexVec3CUDA &P_b,
                          const FlexScalarCUDA &P_m,
                          const FlexScalarCUDA &P_ss,
                          const FlexScalarCUDA &P_s,
                          const FlexScalarCUDA &P_r,
                          const FlexScalarCUDA &P_st,
                          const FlexScalarCUDA &P_ani,
                          const FlexScalarCUDA &P_sh,
                          const FlexScalarCUDA &P_sht,
                          const std::optional<FlexScalarCUDA> &P_c,
                          const std::optional<FlexScalarCUDA> &P_cg,
                          const FlexVec3CUDA &n)
    -> cuda::BRDFInputs {

  cuda::BRDFInputs params;
  params.N = omega_i.shape(0);
  params.omega_i = Vec3ArrayCUDA(omega_i);
  params.omega_o = Vec3ArrayCUDA(omega_o);

  if (requirements.needs_P_b) {
    params.P_b = cuda::broadcast_vec3(P_b, params.N, 0.8F, 0.8F, 0.8F);
  }
  if (requirements.needs_P_m) {
    params.P_m = cuda::broadcast_scalar(P_m, params.N, 0.0F);
  }
  if (requirements.needs_P_ss) {
    params.P_ss = cuda::broadcast_scalar(P_ss, params.N, 0.0F);
  }
  if (requirements.needs_P_s) {
    params.P_s = cuda::broadcast_scalar(P_s, params.N, 0.5F);
  }
  if (requirements.needs_P_r) {
    params.P_r = cuda::broadcast_scalar(P_r, params.N, 0.5F);
  }
  if (requirements.needs_P_st) {
    params.P_st = cuda::broadcast_scalar(P_st, params.N, 0.0F);
  }
  if (requirements.needs_P_ani) {
    params.P_ani = cuda::broadcast_scalar(P_ani, params.N, 0.0F);
  }
  if (requirements.needs_P_sh) {
    params.P_sh = cuda::broadcast_scalar(P_sh, params.N, 0.0F);
  }
  if (requirements.needs_P_sht) {
    params.P_sht = cuda::broadcast_scalar(P_sht, params.N, 0.5F);
  }
  if (requirements.needs_P_c && P_c.has_value()) {
    params.P_c = cuda::broadcast_scalar(*P_c, params.N, 0.0F);
  }
  if (requirements.needs_P_cg && P_cg.has_value()) {
    params.P_cg = cuda::broadcast_scalar(*P_cg, params.N, 1.0F);
  }
  if (requirements.needs_n) {
    params.n = cuda::broadcast_vec3(n, params.N, 0.0F, 0.0F, 1.0F);
  }

  return params;
}

auto inputs_with_defaults(const BRDFInputRequirements &requirements,
                          const Vec3ArrayCPU &omega_i,
                          const Vec3ArrayCPU &omega_o,
                          const FlexVec3CPU &P_b,
                          const FlexScalarCPU &P_m,
                          const FlexScalarCPU &P_ss,
                          const FlexScalarCPU &P_s,
                          const FlexScalarCPU &P_r,
                          const FlexScalarCPU &P_st,
                          const FlexScalarCPU &P_ani,
                          const FlexScalarCPU &P_sh,
                          const FlexScalarCPU &P_sht,
                          const std::optional<FlexScalarCPU> &P_c,
                          const std::optional<FlexScalarCPU> &P_cg, 
                          const FlexVec3CPU &n)
    -> cpu::BRDFInputs {

  cpu::BRDFInputs params;
  params.N = omega_i.shape(0);
  params.omega_i = Vec3ArrayCPU(omega_i);
  params.omega_o = Vec3ArrayCPU(omega_o);
  if (requirements.needs_P_b) {
    params.P_b = cpu::broadcast_vec3(P_b, params.N, 0.8F, 0.8F, 0.8F);
  }
  if (requirements.needs_P_m) {
    params.P_m = cpu::broadcast_scalar(P_m, params.N, 0.0F);
  }
  if (requirements.needs_P_ss) {
    params.P_ss = cpu::broadcast_scalar(P_ss, params.N, 0.0F);
  }
  if (requirements.needs_P_s) {
    params.P_s = cpu::broadcast_scalar(P_s, params.N, 0.5F);
  }
  if (requirements.needs_P_r) {
    params.P_r = cpu::broadcast_scalar(P_r, params.N, 0.5F);
  }
  if (requirements.needs_P_st) {
    params.P_st = cpu::broadcast_scalar(P_st, params.N, 0.0F);
  }
  if (requirements.needs_P_ani) {
    params.P_ani = cpu::broadcast_scalar(P_ani, params.N, 0.0F);
  }
  if (requirements.needs_P_sh) {
    params.P_sh = cpu::broadcast_scalar(P_sh, params.N, 0.0F);
  }
  if (requirements.needs_P_sht) {
    params.P_sht = cpu::broadcast_scalar(P_sht, params.N, 0.5F);
  }
  if (requirements.needs_P_c && P_c.has_value()) {
    params.P_c = cpu::broadcast_scalar(*P_c, params.N, 0.0F);
  }
  if (requirements.needs_P_cg && P_c.has_value()) {
    params.P_cg = cpu::broadcast_scalar(*P_cg, params.N, 1.0F);
  }
  if (requirements.needs_n) {
    params.n = cpu::broadcast_vec3(n, params.N, 0.0F, 0.0F, 1.0F);
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
  BRDFInputRequirements requirements;
  requirements.needs_P_b = true;
  requirements.needs_P_m = true;
  requirements.needs_P_ss = true;
  requirements.needs_P_s = true;
  requirements.needs_P_r = true;
  requirements.needs_P_st = true;
  requirements.needs_P_ani = true;
  requirements.needs_P_sh = true;
  requirements.needs_P_sht = true;
  requirements.needs_P_c = true;
  requirements.needs_P_cg = true;
  requirements.needs_n = true;
  cpu::BRDFInputs params =
      inputs_with_defaults(requirements, omega_i, omega_o, P_b, P_m, P_ss, P_s,
                           P_r, P_st, P_ani, P_sh, P_sht, P_c, P_cg, n);

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

// CUDA version with broadcasting and defaults
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
      cuda::get_common_cuda_device(omega_i, &omega_o, P_b, P_m, P_ss, P_s, P_r,
                                   P_st, P_ani, P_sh, P_sht, P_c, P_cg, n);
  cuda::ScopedCudaDevice device(target_device);

  // Compolete inputs with defaults and broadcast constants for all required
  // parameters
  BRDFInputRequirements requirements;
  requirements.needs_P_b = true;
  requirements.needs_P_m = true;
  requirements.needs_P_ss = true;
  requirements.needs_P_s = true;
  requirements.needs_P_r = true;
  requirements.needs_P_st = true;
  requirements.needs_P_ani = true;
  requirements.needs_P_sh = true;
  requirements.needs_P_sht = true;
  requirements.needs_P_c = true;
  requirements.needs_P_cg = true;
  requirements.needs_n = true;
  cuda::BRDFInputs params =
      inputs_with_defaults(requirements, omega_i, omega_o, P_b, P_m, P_ss, P_s,
                           P_r, P_st, P_ani, P_sh, P_sht, P_c, P_cg, n);

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
  BRDFInputRequirements requirements;
  requirements.needs_P_b = true;
  requirements.needs_P_m = true;
  requirements.needs_P_ss = true;
  requirements.needs_P_s = true;
  requirements.needs_P_r = true;
  requirements.needs_P_st = true;
  requirements.needs_P_ani = true;
  requirements.needs_P_sh = true;
  requirements.needs_P_sht = true;
  requirements.needs_n = true;
  cpu::BRDFInputs params =
      inputs_with_defaults(requirements, omega_i, omega_o, P_b, P_m, P_ss, P_s,
                           P_r, P_st, P_ani, P_sh, P_sht, std::nullopt, std::nullopt, n);

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
  int target_device =
      cuda::get_common_cuda_device(omega_i, &omega_o, P_b, P_m, P_ss, P_s, P_r,
                                   P_st, P_ani, P_sh, P_sht, std::nullopt, std::nullopt, n);
  cuda::ScopedCudaDevice device(target_device);

  // Compolete inputs with defaults and broadcast constant inputs for all required
  // parameters
  BRDFInputRequirements requirements;
  requirements.needs_P_b = true;
  requirements.needs_P_m = true;
  requirements.needs_P_ss = true;
  requirements.needs_P_s = true;
  requirements.needs_P_r = true;
  requirements.needs_P_st = true;
  requirements.needs_P_ani = true;
  requirements.needs_P_sh = true;
  requirements.needs_P_sht = true;
  requirements.needs_n = true;
  cuda::BRDFInputs params =
      inputs_with_defaults(requirements, omega_i, omega_o, P_b, P_m, P_ss, P_s,
                           P_r, P_st, P_ani, P_sh, P_sht, std::nullopt, std::nullopt, n);

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
      "    clearcoat: Clearcoat [N] or [1] (default: 0.0)\n"
      "    clearcoat_gloss: Clearcoat gloss [N] or [1] (default: 1.0)\n"
      "    normal: Surface normal [N, 3] or [1, 3] (default: [0, 0, 1])\n\n"
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
      "    clearcoat: Clearcoat [N] or [1] (default: 0.0)\n"
      "    clearcoat_gloss: Clearcoat gloss [N] or [1] (default: 1.0)\n"
      "    normal: Surface normal [N, 3] or [1, 3] (default: [0, 0, 1])\n\n"
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
      "principled_brdf_backward_P_b", &principled_brdf_backward_P_b_cpu,
      "CPU implementation of the partial derivative of the Principled BRDF "
      "w.r.t. the base color P_b.\n\n"
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
      "principled_brdf_backward_P_b", &principled_brdf_backward_P_b_cuda,
      "GPU implementation of the partial derivative of the Principled BRDF "
      "w.r.t. the base color P_b.\n\n"
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
}
