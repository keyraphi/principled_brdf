#include "../common/common_utils.h"
#include "principled_brdf_cuda.h"
#include <cstddef>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <stdexcept>
#include <math.h>

#define N_THREADS 256

// CUDA kernel
__global__ void add_cuda_kernel(const float *a, const float *b, float *c,
                                size_t n) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    c[idx] = a[idx] + b[idx];
  }
}

// CUDA dummy add implementation
extern "C" void cuda_dummy_add(const float *a, const float *b, float *result,
                               size_t n) {
  // Launch kernel
  add_cuda_kernel<<<(n + 255) / 256, 256>>>(a, b, result, n);

  // Synchronize and check for errors
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    // Clean up on error
    cudaFree(result);
    throw std::runtime_error("CUDA kernel execution failed");
  }

  // Check for kernel launch errors
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    cudaFree(result);
    throw std::runtime_error("CUDA kernel launch failed");
  }
}

__global__ void principled_brdf_forward_kernel(
    const float *__restrict__ omega_i_, const float *__restrict__ omega_o_,
    const float *__restrict__ P_b_, const float *__restrict__ P_m_,
    const float *__restrict__ P_ss_, const float *__restrict__ P_s_,
    const float *__restrict__ P_r_, const float *__restrict__ P_st_,
    const float *__restrict__ P_ani_, const float *__restrict__ P_sh_,
    const float *__restrict__ P_sht_, const float *__restrict__ P_c_,
    const float *__restrict__ P_cg_, const float *__restrict__ n_,
    float *result, size_t N) {
  int global_id = threadIdx.x + blockDim.x * blockIdx.x;

  // shared memory for coalesced reads from global
  extern __shared__ float sh_mem[];

  const int OMEGA_O_OFFSET = 3;
  const int P_b_OFFSET = 6;
  const int N_OFFSET = 9;

  float *s_omega_i = sh_mem;
  float *s_omega_o = sh_mem + (OMEGA_O_OFFSET * N);
  float *s_n = sh_mem + (N_OFFSET * N);
  float *s_P_b = sh_mem + (P_b_OFFSET * N);

#pragma unroll
  for (int i = 0; i < 3; ++i) {
    s_omega_i[threadIdx.x + (i * blockDim.x)] =
        omega_i_[(3 * global_id) + (i * blockDim.x)];
    s_omega_o[threadIdx.x + (i * blockDim.x)] =
        omega_o_[(3 * global_id) + (i * blockDim.x)];
    s_n[threadIdx.x + (i * blockDim.x)] =
        n_[(3 * global_id) + (i * blockDim.x)];
    s_P_b[threadIdx.x + (i * blockDim.x)] =
        P_b_[(3 * global_id) + (i * blockDim.x)];
  }
  __syncthreads();

  // read into registers coalesced and without bank conflicts
  const Vec3 L(s_omega_i[3 * threadIdx.x], s_omega_i[(3 * threadIdx.x) + 1],
               s_omega_i[(3 * threadIdx.x) + 2]);
  const Vec3 V(s_omega_o[3 * threadIdx.x], s_omega_o[(3 * threadIdx.x) + 1],
               s_omega_o[(3 * threadIdx.x) + 2]);
  const Vec3 n(s_n[3 * threadIdx.x], s_n[(3 * threadIdx.x) + 1],
               s_n[(3 * threadIdx.x) + 2]);
  const Vec3 P_b(s_P_b[3 * threadIdx.x], s_P_b[(3 * threadIdx.x) + 1],
                 s_P_b[(3 * threadIdx.x) + 2]);
  const float P_m = P_m_[threadIdx.x];
  const float P_ss = P_ss_[threadIdx.x];
  const float P_s = P_s_[threadIdx.x];
  const float P_r = P_r_[threadIdx.x];
  const float P_st = P_st_[threadIdx.x];
  const float P_ani = P_ani_[threadIdx.x];
  const float P_sh = P_sh_[threadIdx.x];
  const float P_sht = P_sht_[threadIdx.x];
  const float P_c = P_c_[threadIdx.x];
  const float P_cg = P_cg_[threadIdx.x];

  // compute brdf
  const Vec3 H = (V + L).normalize();

  const Vec3 BRDF =
      (M_1_PIf * mix(F_d(V, L, H, n, P_r), ss(V, L, H, n, P_r), P_ss) *
           C_dlin(P_b) +
       F_sheen(L, H, P_b, P_sht, P_sh)) *
          (1.f - P_m) +
      G_s(L, V, n, P_ani, P_r) * F_s(L, H, P_b, P_m, P_st, P_s) *
          D_s(H, n, P_ani, P_r) +
      0.25f * P_c * G_r(L, V, n) * F_r(L, H) * D_r(H, n, P_cg);
  // write out result
  result[(3 * global_id)] = BRDF.x;
  result[(3 * global_id) + 1] = BRDF.y;
  result[(3 * global_id) + 2] = BRDF.z;
}

extern "C" void principled_brdf_cuda_forward(
    const float *__restrict__ omega_i, const float *__restrict__ omega_o,
    const float *__restrict__ P_b, const float *__restrict__ P_m,
    const float *__restrict__ P_ss, const float *__restrict__ P_s,
    const float *__restrict__ P_r, const float *__restrict__ P_st,
    const float *__restrict__ P_ani, const float *__restrict__ P_sh,
    const float *__restrict__ P_sht, const float *__restrict__ P_c,
    const float *__restrict__ P_cg, const float *__restrict__ n, float *result,
    size_t N) {

  const int N_BLOCKS = (N + N_THREADS - 1) / N_THREADS;

  // shared memory for coalesced read in of float3s
  const int shared_memory_bytes = sizeof(float) * 12 * N;
  principled_brdf_forward_kernel<<<N_BLOCKS, N_THREADS, shared_memory_bytes>>>(
      omega_i, omega_o, P_b, P_m, P_ss, P_s, P_r, P_st, P_ani, P_sh, P_sht, P_c,
      P_cg, n, result, N);
  // DEBUGGING
  // Synchronize and check for errors
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    // Clean up on error
    cudaFree(result);
    throw std::runtime_error("CUDA kernel execution failed");
  }

  // Check for kernel launch errors
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    cudaFree(result);
    throw std::runtime_error("CUDA kernel launch failed");
  }
}
