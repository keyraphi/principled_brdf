#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <cstddef>
#include <stdexcept>

#include "cpu/principled_brdf_cpu.h"
#include "cpu/utils.h"
#include "cuda/principled_brdf_cuda.h"
#include "cuda/utils.h"
namespace nb = nanobind;
using namespace nb::literals;

// Device-specific type aliases with exact constraints
using Vec3ArrayCPU = nb::ndarray<const float, nb::shape<-1, 3>, nb::c_contig, nb::device::cpu>;
using ScalarArrayCPU = nb::ndarray<const float, nb::shape<-1>, nb::c_contig, nb::device::cpu>;
using OutputArrayCPU = nb::ndarray<float, nb::shape<-1, 3>, nb::c_contig, nb::device::cpu>;

using Vec3ArrayCUDA = nb::ndarray<const float, nb::shape<-1, 3>, nb::c_contig, nb::device::cuda>;
using ScalarArrayCUDA = nb::ndarray<const float, nb::shape<-1>, nb::c_contig, nb::device::cuda>;
using OutputArrayCUDA = nb::ndarray<float, nb::shape<-1, 3>, nb::c_contig, nb::device::cuda>;

// Flexible input types that allow [1] or [N]
using FlexScalarCPU = nb::ndarray<const float, nb::shape<-1>, nb::c_contig, nb::device::cpu>;
using FlexVec3CPU = nb::ndarray<const float, nb::shape<-1, 3>, nb::c_contig, nb::device::cpu>;

using FlexScalarCUDA = nb::ndarray<const float, nb::shape<-1>, nb::c_contig, nb::device::cuda>;
using FlexVec3CUDA = nb::ndarray<const float, nb::shape<-1, 3>, nb::c_contig, nb::device::cuda>;

// Low-level implementations (no broadcasting, exact shapes [N])
auto principled_brdf_forward_cpu_impl(
    const Vec3ArrayCPU &omega_i,
    const Vec3ArrayCPU &omega_o,
    const Vec3ArrayCPU &P_b,
    const ScalarArrayCPU &P_m,
    const ScalarArrayCPU &P_ss,
    const ScalarArrayCPU &P_s,
    const ScalarArrayCPU &P_r,
    const ScalarArrayCPU &P_st,
    const ScalarArrayCPU &P_ani,
    const ScalarArrayCPU &P_sh,
    const ScalarArrayCPU &P_sht,
    const ScalarArrayCPU &P_c,
    const ScalarArrayCPU &P_cg,
    const Vec3ArrayCPU &n
) -> OutputArrayCPU {
    size_t N = omega_i.shape(0);
    
    float* result_data = new float[N * 3];
    
    principled_brdf_cpu_forward(
        omega_i.data(), omega_o.data(),
        P_b.data(), P_m.data(), P_ss.data(), P_s.data(),
        P_r.data(), P_st.data(), P_ani.data(), P_sh.data(),
        P_sht.data(), P_c.data(), P_cg.data(), n.data(),
        result_data, N
    );
    
    nb::capsule owner(result_data, [](void *ptr) noexcept { delete[] (float*)ptr; });
    return OutputArrayCPU(result_data, {N, 3}, owner);
}

OutputArrayCUDA principled_brdf_forward_cuda_impl(
    const Vec3ArrayCUDA &omega_i,
    const Vec3ArrayCUDA &omega_o,
    const Vec3ArrayCUDA &P_b,
    const ScalarArrayCUDA &P_m,
    const ScalarArrayCUDA &P_ss,
    const ScalarArrayCUDA &P_s,
    const ScalarArrayCUDA &P_r,
    const ScalarArrayCUDA &P_st,
    const ScalarArrayCUDA &P_ani,
    const ScalarArrayCUDA &P_sh,
    const ScalarArrayCUDA &P_sht,
    const ScalarArrayCUDA &P_c,
    const ScalarArrayCUDA &P_cg,
    const Vec3ArrayCUDA &n
) {
    size_t N = omega_i.shape(0);
    
    float* result_data = static_cast<float*>(cuda::cuda_allocate(N * 3 * sizeof(float)));
    if (!result_data) {
        throw std::runtime_error("Failed to allocate CUDA memory for output");
    }
    
    principled_brdf_cuda_forward(
        omega_i.data(), omega_o.data(),
        P_b.data(), P_m.data(), P_ss.data(), P_s.data(),
        P_r.data(), P_st.data(), P_ani.data(), P_sh.data(),
        P_sht.data(), P_c.data(), P_cg.data(), n.data(),
        result_data, N
    );
    
    nb::capsule owner(result_data, [](void *p) noexcept { cuda::cuda_free(p); });
    return OutputArrayCUDA(result_data, {N, 3}, owner);
}

// CPU version with broadcasting and defaults
OutputArrayCPU principled_brdf_forward_cpu(
    const Vec3ArrayCPU &omega_i,
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
    const FlexVec3CPU &n = FlexVec3CPU()
) {
    size_t N = omega_i.shape(0);
    
    // Create defaults if needed using CPU utilities
    Vec3ArrayCPU P_b_final = P_b.size() > 0 ? 
        cpu::broadcast_vec3(P_b, N) : cpu::create_default_vec3(N, 0.8f, 0.8f, 0.8f);
    ScalarArrayCPU P_m_final = P_m.size() > 0 ? 
        cpu::broadcast_scalar(P_m, N) : cpu::create_default_scalar(N, 0.0f);
    ScalarArrayCPU P_ss_final = P_ss.size() > 0 ? 
        cpu::broadcast_scalar(P_ss, N) : cpu::create_default_scalar(N, 0.0f);
    ScalarArrayCPU P_s_final = P_s.size() > 0 ? 
        cpu::broadcast_scalar(P_s, N) : cpu::create_default_scalar(N, 0.5f);
    ScalarArrayCPU P_r_final = P_r.size() > 0 ? 
        cpu::broadcast_scalar(P_r, N) : cpu::create_default_scalar(N, 0.5f);
    ScalarArrayCPU P_st_final = P_st.size() > 0 ? 
        cpu::broadcast_scalar(P_st, N) : cpu::create_default_scalar(N, 0.0f);
    ScalarArrayCPU P_ani_final = P_ani.size() > 0 ? 
        cpu::broadcast_scalar(P_ani, N) : cpu::create_default_scalar(N, 0.0f);
    ScalarArrayCPU P_sh_final = P_sh.size() > 0 ? 
        cpu::broadcast_scalar(P_sh, N) : cpu::create_default_scalar(N, 0.0f);
    ScalarArrayCPU P_sht_final = P_sht.size() > 0 ? 
        cpu::broadcast_scalar(P_sht, N) : cpu::create_default_scalar(N, 0.5f);
    ScalarArrayCPU P_c_final = P_c.size() > 0 ? 
        cpu::broadcast_scalar(P_c, N) : cpu::create_default_scalar(N, 0.0f);
    ScalarArrayCPU P_cg_final = P_cg.size() > 0 ? 
        cpu::broadcast_scalar(P_cg, N) : cpu::create_default_scalar(N, 1.0f);
    Vec3ArrayCPU n_final = n.size() > 0 ? 
        cpu::broadcast_vec3(n, N) : cpu::create_default_vec3(N, 0.0f, 0.0f, 1.0f);
    
    return principled_brdf_forward_cpu_impl(
        omega_i, omega_o,
        P_b_final, P_m_final, P_ss_final, P_s_final,
        P_r_final, P_st_final, P_ani_final, P_sh_final,
        P_sht_final, P_c_final, P_cg_final, n_final
    );
}

// CUDA version with broadcasting and defaults
OutputArrayCUDA principled_brdf_forward_cuda(
    const Vec3ArrayCUDA &omega_i,
    const Vec3ArrayCUDA &omega_o,
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
    const FlexVec3CUDA &n = FlexVec3CUDA()
) {
    size_t N = omega_i.shape(0);
    
    // TODO this here is doing double work - fix that.
    // Create defaults if needed using CUDA utilities
    Vec3ArrayCUDA P_b_final = 
        cuda::broadcast_vec3(P_b, N, 0.8f, 0.8f, 0.8f);
    ScalarArrayCUDA P_m_final = 
        cuda::broadcast_scalar(P_m, N, 0.0f);
    ScalarArrayCUDA P_ss_final =
        cuda::broadcast_scalar(P_ss, N, 0.0f);
    ScalarArrayCUDA P_s_final =
        cuda::broadcast_scalar(P_s, N, 0.5f);
    ScalarArrayCUDA P_r_final =
        cuda::broadcast_scalar(P_r, N, 0.5f);
    ScalarArrayCUDA P_st_final =
        cuda::broadcast_scalar(P_st, N, 0.0f);
    ScalarArrayCUDA P_ani_final =
        cuda::broadcast_scalar(P_ani, N, 0.0f);
    ScalarArrayCUDA P_sh_final =
        cuda::broadcast_scalar(P_sh, N, 0.0f);
    ScalarArrayCUDA P_sht_final =
        cuda::broadcast_scalar(P_sht, N, 0.5f);
    ScalarArrayCUDA P_c_final = 
        cuda::broadcast_scalar(P_c, N, 0.0f);
    ScalarArrayCUDA P_cg_final =
        cuda::broadcast_scalar(P_cg, N, 1.0f);
    Vec3ArrayCUDA n_final =
        cuda::broadcast_vec3(n, N, 0.0f, 0.0f, 1.0f);
    
    return principled_brdf_forward_cuda_impl(
        Vec3ArrayCUDA(omega_i), Vec3ArrayCUDA(omega_o),
        P_b_final, P_m_final, P_ss_final, P_s_final,
        P_r_final, P_st_final, P_ani_final, P_sh_final,
        P_sht_final, P_c_final, P_cg_final, n_final
    );
}

nb::ndarray<float> dummy_add(const nb::ndarray<float> &a,
                             const nb::ndarray<float> &b) {
  if (a.shape(0) != b.shape(0) || a.shape(1) != b.shape(1)) {
    throw std::runtime_error("Shape mismatch");
  }

  size_t n = a.shape(0) * a.shape(1);
  size_t shape[2] = {a.shape(0), a.shape(1)};

  if (a.device_type() == nb::device::cpu::value) {
    // Allocate CPU memory
    float *result_data = new float[n];

    // CPU implementation
    const float *a_data = a.data();
    const float *b_data = b.data();

    for (size_t i = 0; i < n; ++i) {
      result_data[i] = a_data[i] + b_data[i];
    }

    // Create ndarray that takes ownership of the memory
    nb::capsule owner(result_data,
                      [](void *p) noexcept { delete[] (float *)p; });

    return nb::ndarray<float>(result_data, 2, shape, owner);
  } 
  if (a.device_type() == nb::device::cuda::value) {
    // Allocate CUDA memory using our CUDA function
    auto *result_data = static_cast<float *>(cuda::cuda_allocate(n * sizeof(float)));
    if (!result_data) {
      throw std::runtime_error("Failed to allocate CUDA memory");
    }

    // Call CUDA implementation
    const float *a_data = a.data();
    const float *b_data = b.data();
    cuda_dummy_add(a_data, b_data, result_data, n);

    // Create ndarray that takes ownership of the CUDA memory
    nb::capsule owner(result_data, [](void *p) noexcept { cuda::cuda_free(p); });

    return nb::ndarray<float>(result_data, 2, shape, owner,
                              nullptr, // strides (nullptr = contiguous)
                              nb::dtype<float>(), nb::device::cuda::value,
                              a.device_id());
  } 
  throw std::runtime_error("Unsupported device type");
}

NB_MODULE(principled_brdf_functions, m) {
    m.doc() = "Raw Principled BRDF functions with partial derivatives (C++/CUDA implementation)";
    
    m.def("dummy_add", &dummy_add, "Add two arrays");
    
    // Two separate overloads - nanobind will automatically dispatch based on device type
    m.def("principled_brdf_forward", &principled_brdf_forward_cpu,
          "CPU implementatio of the Principled BRDF forward pass.\n\n"
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
          "omega_i"_a, "omega_o"_a,
          nb::kw_only(),
          "basecolor"_a = FlexVec3CPU(), "metallic"_a = FlexScalarCPU(),
          "subsurface"_a = FlexScalarCPU(), "specular"_a = FlexScalarCPU(),
          "roughness"_a = FlexScalarCPU(), "specular_tint"_a = FlexScalarCPU(),
          "anisotropy"_a = FlexScalarCPU(), "sheen"_a = FlexScalarCPU(),
          "sheen_tint"_a = FlexScalarCPU(), "clearcoat"_a = FlexScalarCPU(),
          "clearcoat_gloss"_a = FlexScalarCPU(), "normal"_a = FlexVec3CPU()
    );
    
    m.def("principled_brdf_forward", &principled_brdf_forward_cuda,
          "GPU implementatio of the Principled BRDF forward pass.\n\n"
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
          "omega_i"_a, "omega_o"_a,
          nb::kw_only(),
          "basecolor"_a = FlexVec3CUDA(), "metallic"_a = FlexScalarCUDA(),
          "subsurface"_a = FlexScalarCUDA(), "specular"_a = FlexScalarCUDA(),
          "roughness"_a = FlexScalarCUDA(), "specular_tint"_a = FlexScalarCUDA(),
          "anisotropy"_a = FlexScalarCUDA(), "sheen"_a = FlexScalarCUDA(),
          "sheen_tint"_a = FlexScalarCUDA(), "clearcoat"_a = FlexScalarCUDA(),
          "clearcoat_gloss"_a = FlexScalarCUDA(), "normal"_a = FlexVec3CUDA()
    );
}
