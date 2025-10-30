#include "../common/brdf_utils.h"
#include "principled_brdf_cuda.h"
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <math.h>
#include <stdexcept>

#define N_THREADS 256

__global__ void principled_brdf_forward_kernel(
    const float *__restrict__ omega_i_, const float *__restrict__ omega_o_,
    const float *__restrict__ P_b_, const float *__restrict__ P_m_,
    const float *__restrict__ P_ss_, const float *__restrict__ P_s_,
    const float *__restrict__ P_r_, const float *__restrict__ P_st_,
    const float *__restrict__ P_ani_, const float *__restrict__ P_sh_,
    const float *__restrict__ P_sht_, const float *__restrict__ P_c_,
    const float *__restrict__ P_cg_, const float *__restrict__ n_,
    float *result, size_t N) {
  uint32_t global_id = threadIdx.x + blockDim.x * blockIdx.x;

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

extern "C" void principled_brdf_forward_cuda_impl(
    const float *__restrict__ omega_i, const float *__restrict__ omega_o,
    const float *__restrict__ P_b, const float *__restrict__ P_m,
    const float *__restrict__ P_ss, const float *__restrict__ P_s,
    const float *__restrict__ P_r, const float *__restrict__ P_st,
    const float *__restrict__ P_ani, const float *__restrict__ P_sh,
    const float *__restrict__ P_sht, const float *__restrict__ P_c,
    const float *__restrict__ P_cg, const float *__restrict__ n, float *result,
    size_t N) {

  const size_t N_BLOCKS = (N + N_THREADS - 1) / N_THREADS;

  // shared memory for coalesced read in of float3s
  const size_t shared_memory_bytes = sizeof(float) * 12 * N;
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

__global__ void principled_brdf_backward_P_b_kernel(
    const float *__restrict__ omega_i_, const float *__restrict__ omega_o_,
    const float *__restrict__ P_b_, const float *__restrict__ P_m_,
    const float *__restrict__ P_ss_, const float *__restrict__ P_s_,
    const float *__restrict__ P_r_, const float *__restrict__ P_st_,
    const float *__restrict__ P_ani_, const float *__restrict__ P_sh_,
    const float *__restrict__ P_sht_, const float *__restrict__ n_,
    float *result, size_t N) {
  uint32_t global_id = threadIdx.x + blockDim.x * blockIdx.x;

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

  // derivative
  const Vec3 H = (V + L).normalize();

  const Mat3x3 dBRDFdP_b =
      ((1.F - P_m) * M_1_PIf *
       mix(F_d(V, L, H, n, P_r), ss(V, L, H, n, P_r), P_ss) *
       dC_dlin_dP_b(P_b)) +
      dF_sheen_dP_b(L, H, P_b, P_sh, P_sht) +
      (D_s(H, n, P_ani, P_r) * G_s(L, V, n, P_ani, P_r) *
       dF_s_dP_b(L, H, P_b, P_s, P_st, P_m));

  result[0] = dBRDFdP_b.m[0];
  result[1] = dBRDFdP_b.m[1];
  result[2] = dBRDFdP_b.m[2];
  result[3] = dBRDFdP_b.m[3];
  result[4] = dBRDFdP_b.m[4];
  result[5] = dBRDFdP_b.m[5];
  result[6] = dBRDFdP_b.m[6];
  result[7] = dBRDFdP_b.m[7];
  result[8] = dBRDFdP_b.m[8];
}

extern "C" void principled_brdf_backward_P_b_cuda_impl(
    const float *__restrict__ omega_i, const float *__restrict__ omega_o,
    const float *__restrict__ P_b, const float *__restrict__ P_m,
    const float *__restrict__ P_ss, const float *__restrict__ P_s,
    const float *__restrict__ P_r, const float *__restrict__ P_st,
    const float *__restrict__ P_ani, const float *__restrict__ P_sh,
    const float *__restrict__ P_sht, const float *__restrict__ n, float *result,
    size_t N) {

  const size_t N_BLOCKS = (N + N_THREADS - 1) / N_THREADS;

  // shared memory for coalesced read in of float3s
  const size_t shared_memory_bytes = sizeof(float) * 12 * N;
  principled_brdf_backward_P_b_kernel<<<N_BLOCKS, N_THREADS,
                                        shared_memory_bytes>>>(
      omega_i, omega_o, P_b, P_m, P_ss, P_s, P_r, P_st, P_ani, P_sh, P_sht, n,
      result, N);

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

__global__ void principled_brdf_backward_P_m_kernel(
    const float *__restrict__ omega_i_, const float *__restrict__ omega_o_,
    const float *__restrict__ P_b_, const float *__restrict__ P_ss_,
    const float *__restrict__ P_s_, const float *__restrict__ P_r_,
    const float *__restrict__ P_st_, const float *__restrict__ P_ani_,
    const float *__restrict__ P_sh_, const float *__restrict__ P_sht_,
    const float *__restrict__ n_, float *result, size_t N) {
  uint32_t global_id = threadIdx.x + blockDim.x * blockIdx.x;

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
  const float P_ss = P_ss_[threadIdx.x];
  const float P_s = P_s_[threadIdx.x];
  const float P_r = P_r_[threadIdx.x];
  const float P_st = P_st_[threadIdx.x];
  const float P_ani = P_ani_[threadIdx.x];
  const float P_sh = P_sh_[threadIdx.x];
  const float P_sht = P_sht_[threadIdx.x];

  // derivative
  const Vec3 H = (V + L).normalize();

  const Vec3 dBRDFdP_m =
      -(M_1_PIf * mix(F_d(V, L, H, n, P_r), ss(V, L, H, n, P_r), P_ss) *
            C_dlin(P_b) +
        F_sheen(L, H, P_b, P_sht, P_sh)) +
      G_s(L, V, n, P_ani, P_r) * D_s(H, n, P_ani, P_r) *
          dF_s_dP_m(L, H, P_b, P_st, P_s);

  result[0] = dBRDFdP_m.x;
  result[1] = dBRDFdP_m.y;
  result[2] = dBRDFdP_m.z;
}

extern "C" void principled_brdf_backward_P_m_cuda_impl(
    const float *__restrict__ omega_i, const float *__restrict__ omega_o,
    const float *__restrict__ P_b, const float *__restrict__ P_ss,
    const float *__restrict__ P_s, const float *__restrict__ P_r,
    const float *__restrict__ P_st, const float *__restrict__ P_ani,
    const float *__restrict__ P_sh, const float *__restrict__ P_sht,
    const float *__restrict__ n, float *__restrict__ result, size_t N) {

  const size_t N_BLOCKS = (N + N_THREADS - 1) / N_THREADS;

  // shared memory for coalesced read in of float3s
  const size_t shared_memory_bytes = sizeof(float) * 12 * N;
  principled_brdf_backward_P_m_kernel<<<N_BLOCKS, N_THREADS,
                                        shared_memory_bytes>>>(
      omega_i, omega_o, P_b, P_ss, P_s, P_r, P_st, P_ani, P_sh, P_sht, n,
      result, N);

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

__global__ void principled_brdf_backward_P_ss_kernel(
    const float *__restrict__ omega_i_, const float *__restrict__ omega_o_,
    const float *__restrict__ P_b_, const float *__restrict__ P_m_,
    const float *__restrict__ P_r_, const float *__restrict__ n_,
    float *__restrict__ result, size_t N) {
  uint32_t global_id = threadIdx.x + blockDim.x * blockIdx.x;

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
  const float P_r = P_r_[threadIdx.x];

  // derivative
  const Vec3 H = (V + L).normalize();

  const Vec3 dBRDF_dP_ss = (1.F - P_m) * M_1_PIf * C_dlin(P_b) *
                           (ss(V, L, H, n, P_r) - F_d(V, L, H, n, P_r));

  result[0] = dBRDF_dP_ss.x;
  result[1] = dBRDF_dP_ss.y;
  result[2] = dBRDF_dP_ss.z;
}

extern "C" void principled_brdf_backward_P_ss_cuda_impl(
    const float *__restrict__ omega_i, const float *__restrict__ omega_o,
    const float *__restrict__ P_b, const float *__restrict__ P_m,
    const float *__restrict__ P_r, const float *__restrict__ n,
    float *__restrict__ result, size_t N) {

  const size_t N_BLOCKS = (N + N_THREADS - 1) / N_THREADS;

  // shared memory for coalesced read in of float3s
  const size_t shared_memory_bytes = sizeof(float) * 12 * N;
  principled_brdf_backward_P_ss_kernel<<<N_BLOCKS, N_THREADS,
                                         shared_memory_bytes>>>(
      omega_i, omega_o, P_b, P_m, P_r, n, result, N);

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

__global__ void principled_brdf_backward_P_s_kernel(
    const float *__restrict__ omega_i_, const float *__restrict__ omega_o_,
    const float *__restrict__ P_b_, const float *__restrict__ P_m_,
    const float *__restrict__ P_st_, const float *__restrict__ n_,
    float *__restrict__ result, size_t N) {
  uint32_t global_id = threadIdx.x + blockDim.x * blockIdx.x;

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
  const float P_st = P_st_[threadIdx.x];

  // compute brdf
  const Vec3 H = (V + L).normalize();

  const Vec3 dBRDF_dP_s = (1.F - F_H(L, H)) * dC_spec0_dP_s(P_b, P_m, P_st);
  // write out result
  result[(3 * global_id)] = dBRDF_dP_s.x;
  result[(3 * global_id) + 1] = dBRDF_dP_s.y;
  result[(3 * global_id) + 2] = dBRDF_dP_s.z;
}

extern "C" void principled_brdf_backward_P_s_cuda_impl(
    const float *__restrict__ omega_i, const float *__restrict__ omega_o,
    const float *__restrict__ P_b, const float *__restrict__ P_m,
    const float *__restrict__ P_st, const float *__restrict__ n,
    float *__restrict__ result, size_t N) {

  const size_t N_BLOCKS = (N + N_THREADS - 1) / N_THREADS;

  // shared memory for coalesced read in of float3s
  const size_t shared_memory_bytes = sizeof(float) * 12 * N;
  principled_brdf_backward_P_s_kernel<<<N_BLOCKS, N_THREADS,
                                        shared_memory_bytes>>>(
      omega_i, omega_o, P_b, P_m, P_st, n, result, N);
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

__global__ void principled_brdf_backward_P_r_kernel(
    const float *__restrict__ omega_i_, const float *__restrict__ omega_o_,
    const float *__restrict__ P_b_, const float *__restrict__ P_m_,
    const float *__restrict__ P_ss_, const float *__restrict__ P_s_,
    const float *__restrict__ P_r_, const float *__restrict__ P_st_,
    const float *__restrict__ P_ani_, const float *__restrict__ n_,
    float *result, size_t N) {
  uint32_t global_id = threadIdx.x + blockDim.x * blockIdx.x;

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

  // compute brdf
  const Vec3 H = (V + L).normalize();

  const Vec3 dBRDF_dP_r =
      (1.F - P_m) * M_1_PIf * C_dlin(P_b) *
          ((1.F - P_ss) * dF_d_dP_r(L, V, H, P_r, n) +
           dss_dP_r(L, V, H, P_r, n)) +
      F_s(L, H, P_b, P_m, P_st, P_s) *
          (dG_s_dP_r(L, V, P_r, P_ani, n) * D_s(H, n, P_ani, P_r) +
           G_s(L, V, n, P_ani, P_r) * dD_s_dP_r(H, P_r, P_ani, n));
  // write out result
  result[(3 * global_id)] = dBRDF_dP_r.x;
  result[(3 * global_id) + 1] = dBRDF_dP_r.y;
  result[(3 * global_id) + 2] = dBRDF_dP_r.z;
}

extern "C" void principled_brdf_backward_P_r_cuda_impl(
    const float *__restrict__ omega_i, const float *__restrict__ omega_o,
    const float *__restrict__ P_b, const float *__restrict__ P_m,
    const float *__restrict__ P_ss, const float *__restrict__ P_s,
    const float *__restrict__ P_r, const float *__restrict__ P_st,
    const float *__restrict__ P_ani, const float *__restrict__ n, float *result,
    size_t N) {

  const size_t N_BLOCKS = (N + N_THREADS - 1) / N_THREADS;

  // shared memory for coalesced read in of float3s
  const size_t shared_memory_bytes = sizeof(float) * 12 * N;
  principled_brdf_backward_P_r_kernel<<<N_BLOCKS, N_THREADS,
                                        shared_memory_bytes>>>(
      omega_i, omega_o, P_b, P_m, P_ss, P_s, P_r, P_st, P_ani, n, result, N);
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

__global__ void principled_brdf_backward_P_st_kernel(
    const float *__restrict__ omega_i_, const float *__restrict__ omega_o_,
    const float *__restrict__ P_b_, const float *__restrict__ P_m_,
    const float *__restrict__ P_s_, const float *__restrict__ P_r_,
    const float *__restrict__ P_ani_, const float *__restrict__ n_,
    float *result, size_t N) {
  uint32_t global_id = threadIdx.x + blockDim.x * blockIdx.x;

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
  const float P_s = P_s_[threadIdx.x];
  const float P_r = P_r_[threadIdx.x];
  const float P_ani = P_ani_[threadIdx.x];

  // compute brdf
  const Vec3 H = (V + L).normalize();

  const Vec3 dBRDF_dP_st = G_s(L, V, n, P_ani, P_r) * D_s(H, n, P_ani, P_r) *
                           dF_s_dP_st(L, H, P_b, P_m, P_s);

  // write out result
  result[(3 * global_id)] = dBRDF_dP_st.x;
  result[(3 * global_id) + 1] = dBRDF_dP_st.y;
  result[(3 * global_id) + 2] = dBRDF_dP_st.z;
}

extern "C" void principled_brdf_backward_P_st_cuda_impl(
    const float *__restrict__ omega_i, const float *__restrict__ omega_o,
    const float *__restrict__ P_b, const float *__restrict__ P_m,
    const float *__restrict__ P_s, const float *__restrict__ P_r,
    const float *__restrict__ P_ani, const float *__restrict__ n,
    float *__restrict__ result, size_t N) {

  const size_t N_BLOCKS = (N + N_THREADS - 1) / N_THREADS;

  // shared memory for coalesced read in of float3s
  const size_t shared_memory_bytes = sizeof(float) * 12 * N;
  principled_brdf_backward_P_st_kernel<<<N_BLOCKS, N_THREADS,
                                         shared_memory_bytes>>>(
      omega_i, omega_o, P_b, P_m, P_s, P_r, P_ani, n, result, N);
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

__global__ void principled_brdf_backward_P_ani_kernel(
    const float *__restrict__ omega_i_, const float *__restrict__ omega_o_,
    const float *__restrict__ P_b_, const float *__restrict__ P_m_,
    const float *__restrict__ P_s_, const float *__restrict__ P_r_,
    const float *__restrict__ P_st_, const float *__restrict__ P_ani_,
    const float *__restrict__ n_, float *result, size_t N) {
  uint32_t global_id = threadIdx.x + blockDim.x * blockIdx.x;

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
  const float P_s = P_s_[threadIdx.x];
  const float P_r = P_r_[threadIdx.x];
  const float P_st = P_st_[threadIdx.x];
  const float P_ani = P_ani_[threadIdx.x];

  // compute brdf
  const Vec3 H = (V + L).normalize();

  const Vec3 dBRDF_dP_ani =
      dBRDF_da_x(L, V, H, P_b, P_m, P_st, P_s, P_ani, P_r, n) *
          da_x_daspect(P_r, P_ani) * daspect_dP_ani(P_ani) +
      dBRDF_da_y(L, V, H, P_b, P_m, P_st, P_s, P_ani, P_r, n) *
          da_y_daspect(P_r) * daspect_dP_ani(P_ani);
  // write out result
  result[(3 * global_id)] = dBRDF_dP_ani.x;
  result[(3 * global_id) + 1] = dBRDF_dP_ani.y;
  result[(3 * global_id) + 2] = dBRDF_dP_ani.z;
}

extern "C" void principled_brdf_backward_P_ani_cuda_impl(
    const float *__restrict__ omega_i, const float *__restrict__ omega_o,
    const float *__restrict__ P_b, const float *__restrict__ P_m,
    const float *__restrict__ P_s, const float *__restrict__ P_r,
    const float *__restrict__ P_st, const float *__restrict__ P_ani,
    const float *__restrict__ n, float *__restrict__ result, size_t N) {

  const size_t N_BLOCKS = (N + N_THREADS - 1) / N_THREADS;

  // shared memory for coalesced read in of float3s
  const size_t shared_memory_bytes = sizeof(float) * 12 * N;
  principled_brdf_backward_P_ani_kernel<<<N_BLOCKS, N_THREADS,
                                          shared_memory_bytes>>>(
      omega_i, omega_o, P_b, P_m, P_s, P_r, P_st, P_ani, n, result, N);
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

__global__ void principled_brdf_backward_P_sh_kernel(
    const float *__restrict__ omega_i_, const float *__restrict__ omega_o_,
    const float *__restrict__ P_b_, const float *__restrict__ P_m_,
    const float *__restrict__ P_sht_, float *__restrict__ result, size_t N) {
  uint32_t global_id = threadIdx.x + blockDim.x * blockIdx.x;

  // shared memory for coalesced reads from global
  extern __shared__ float sh_mem[];

  const int OMEGA_O_OFFSET = 3;
  const int P_b_OFFSET = 6;

  float *s_omega_i = sh_mem;
  float *s_omega_o = sh_mem + (OMEGA_O_OFFSET * N);
  float *s_P_b = sh_mem + (P_b_OFFSET * N);

#pragma unroll
  for (int i = 0; i < 3; ++i) {
    s_omega_i[threadIdx.x + (i * blockDim.x)] =
        omega_i_[(3 * global_id) + (i * blockDim.x)];
    s_omega_o[threadIdx.x + (i * blockDim.x)] =
        omega_o_[(3 * global_id) + (i * blockDim.x)];
    s_P_b[threadIdx.x + (i * blockDim.x)] =
        P_b_[(3 * global_id) + (i * blockDim.x)];
  }
  __syncthreads();

  // read into registers coalesced and without bank conflicts
  const Vec3 L(s_omega_i[3 * threadIdx.x], s_omega_i[(3 * threadIdx.x) + 1],
               s_omega_i[(3 * threadIdx.x) + 2]);
  const Vec3 V(s_omega_o[3 * threadIdx.x], s_omega_o[(3 * threadIdx.x) + 1],
               s_omega_o[(3 * threadIdx.x) + 2]);
  const Vec3 P_b(s_P_b[3 * threadIdx.x], s_P_b[(3 * threadIdx.x) + 1],
                 s_P_b[(3 * threadIdx.x) + 2]);
  const float P_m = P_m_[threadIdx.x];
  const float P_sht = P_sht_[threadIdx.x];

  // compute brdf
  const Vec3 H = (V + L).normalize();

  const Vec3 dBRDF_dP_sh = (1.F - P_m) * dF_sheen_dP_sh(L, H, P_b, P_sht);
  // write out result
  result[(3 * global_id)] = dBRDF_dP_sh.x;
  result[(3 * global_id) + 1] = dBRDF_dP_sh.y;
  result[(3 * global_id) + 2] = dBRDF_dP_sh.z;
}

extern "C" void principled_brdf_backward_P_sh_cuda_impl(
    const float *__restrict__ omega_i, const float *__restrict__ omega_o,
    const float *__restrict__ P_b, const float *__restrict__ P_m,
    const float *__restrict__ P_sht, float *__restrict__ result, size_t N) {

  const size_t N_BLOCKS = (N + N_THREADS - 1) / N_THREADS;

  // shared memory for coalesced read in of float3s
  const size_t shared_memory_bytes = sizeof(float) * 9 * N;
  principled_brdf_backward_P_sh_kernel<<<N_BLOCKS, N_THREADS,
                                         shared_memory_bytes>>>(
      omega_i, omega_o, P_b, P_m, P_sht, result, N);
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

__global__ void principled_brdf_backward_P_sht_kernel(
    const float *__restrict__ omega_i_, const float *__restrict__ omega_o_,
    const float *__restrict__ P_b_, const float *__restrict__ P_m_,
    const float *__restrict__ P_sh_, float *__restrict__ result, size_t N) {
  uint32_t global_id = threadIdx.x + blockDim.x * blockIdx.x;

  // shared memory for coalesced reads from global
  extern __shared__ float sh_mem[];

  const int OMEGA_O_OFFSET = 3;
  const int P_b_OFFSET = 6;

  float *s_omega_i = sh_mem;
  float *s_omega_o = sh_mem + (OMEGA_O_OFFSET * N);
  float *s_P_b = sh_mem + (P_b_OFFSET * N);

#pragma unroll
  for (int i = 0; i < 3; ++i) {
    s_omega_i[threadIdx.x + (i * blockDim.x)] =
        omega_i_[(3 * global_id) + (i * blockDim.x)];
    s_omega_o[threadIdx.x + (i * blockDim.x)] =
        omega_o_[(3 * global_id) + (i * blockDim.x)];
    s_P_b[threadIdx.x + (i * blockDim.x)] =
        P_b_[(3 * global_id) + (i * blockDim.x)];
  }
  __syncthreads();

  // read into registers coalesced and without bank conflicts
  const Vec3 L(s_omega_i[3 * threadIdx.x], s_omega_i[(3 * threadIdx.x) + 1],
               s_omega_i[(3 * threadIdx.x) + 2]);
  const Vec3 V(s_omega_o[3 * threadIdx.x], s_omega_o[(3 * threadIdx.x) + 1],
               s_omega_o[(3 * threadIdx.x) + 2]);
  const Vec3 P_b(s_P_b[3 * threadIdx.x], s_P_b[(3 * threadIdx.x) + 1],
                 s_P_b[(3 * threadIdx.x) + 2]);
  const float P_m = P_m_[threadIdx.x];
  const float P_sh = P_sh_[threadIdx.x];

  // compute brdf
  const Vec3 H = (V + L).normalize();

  const Vec3 dBRDF_dP_sht = (1.F - P_m) * dF_sheen_dP_sht(L, H, P_b, P_sh);
  // write out result
  result[(3 * global_id)] = dBRDF_dP_sht.x;
  result[(3 * global_id) + 1] = dBRDF_dP_sht.y;
  result[(3 * global_id) + 2] = dBRDF_dP_sht.z;
}

extern "C" void principled_brdf_backward_P_sht_cuda_impl(
    const float *__restrict__ omega_i, const float *__restrict__ omega_o,
    const float *__restrict__ P_b, const float *__restrict__ P_m,
    const float *__restrict__ P_sh, float *__restrict__ result, size_t N) {

  const size_t N_BLOCKS = (N + N_THREADS - 1) / N_THREADS;

  // shared memory for coalesced read in of float3s
  const size_t shared_memory_bytes = sizeof(float) * 9 * N;
  principled_brdf_backward_P_sht_kernel<<<N_BLOCKS, N_THREADS,
                                          shared_memory_bytes>>>(
      omega_i, omega_o, P_b, P_m, P_sh, result, N);
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
__global__ void principled_brdf_backward_P_c_kernel(
    const float *__restrict__ omega_i_, const float *__restrict__ omega_o_,
    const float *__restrict__ P_cg_, const float *__restrict__ n_,
    float *__restrict__ result, size_t N) {
  uint32_t global_id = threadIdx.x + blockDim.x * blockIdx.x;

  // shared memory for coalesced reads from global
  extern __shared__ float sh_mem[];

  const int OMEGA_O_OFFSET = 3;
  const int N_OFFSET = 6;

  float *s_omega_i = sh_mem;
  float *s_omega_o = sh_mem + (OMEGA_O_OFFSET * N);
  float *s_n = sh_mem + (N_OFFSET * N);

#pragma unroll
  for (int i = 0; i < 3; ++i) {
    s_omega_i[threadIdx.x + (i * blockDim.x)] =
        omega_i_[(3 * global_id) + (i * blockDim.x)];
    s_omega_o[threadIdx.x + (i * blockDim.x)] =
        omega_o_[(3 * global_id) + (i * blockDim.x)];
    s_n[threadIdx.x + (i * blockDim.x)] =
        n_[(3 * global_id) + (i * blockDim.x)];
  }
  __syncthreads();

  // read into registers coalesced and without bank conflicts
  const Vec3 L(s_omega_i[3 * threadIdx.x], s_omega_i[(3 * threadIdx.x) + 1],
               s_omega_i[(3 * threadIdx.x) + 2]);
  const Vec3 V(s_omega_o[3 * threadIdx.x], s_omega_o[(3 * threadIdx.x) + 1],
               s_omega_o[(3 * threadIdx.x) + 2]);
  const Vec3 n(s_n[3 * threadIdx.x], s_n[(3 * threadIdx.x) + 1],
               s_n[(3 * threadIdx.x) + 2]);
  const float P_cg = P_cg_[threadIdx.x];

  // compute brdf
  const Vec3 H = (V + L).normalize();

  // Multiply 1 vector - each color channel gets same derivative
  const float dBRDF_dP_c = 0.25F * G_r(L, V, n) * F_r(L, H) * D_r(H, n, P_cg);
  // write out result
  result[(3 * global_id)] = dBRDF_dP_c;
  result[(3 * global_id) + 1] = dBRDF_dP_c;
  result[(3 * global_id) + 2] = dBRDF_dP_c;
}

extern "C" void principled_brdf_backward_P_c_cuda_impl(
    const float *__restrict__ omega_i, const float *__restrict__ omega_o,
    const float *__restrict__ P_cg, const float *__restrict__ n,
    float *__restrict__ result, size_t N) {

  const size_t N_BLOCKS = (N + N_THREADS - 1) / N_THREADS;

  // shared memory for coalesced read in of float3s
  const size_t shared_memory_bytes = sizeof(float) * 9 * N;
  principled_brdf_backward_P_c_kernel<<<N_BLOCKS, N_THREADS,
                                        shared_memory_bytes>>>(
      omega_i, omega_o, P_cg, n, result, N);
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

__global__ void principled_brdf_backward_P_cg_kernel(
    const float *__restrict__ omega_i_, const float *__restrict__ omega_o_,
    const float *__restrict__ P_c_, const float *__restrict__ P_cg_,
    const float *__restrict__ n_, float *__restrict__ result, size_t N) {
  uint32_t global_id = threadIdx.x + blockDim.x * blockIdx.x;

  // shared memory for coalesced reads from global
  extern __shared__ float sh_mem[];

  const int OMEGA_O_OFFSET = 3;
  const int N_OFFSET = 6;

  float *s_omega_i = sh_mem;
  float *s_omega_o = sh_mem + (OMEGA_O_OFFSET * N);
  float *s_n = sh_mem + (N_OFFSET * N);

#pragma unroll
  for (int i = 0; i < 3; ++i) {
    s_omega_i[threadIdx.x + (i * blockDim.x)] =
        omega_i_[(3 * global_id) + (i * blockDim.x)];
    s_omega_o[threadIdx.x + (i * blockDim.x)] =
        omega_o_[(3 * global_id) + (i * blockDim.x)];
    s_n[threadIdx.x + (i * blockDim.x)] =
        n_[(3 * global_id) + (i * blockDim.x)];
  }
  __syncthreads();

  // read into registers coalesced and without bank conflicts
  const Vec3 L(s_omega_i[3 * threadIdx.x], s_omega_i[(3 * threadIdx.x) + 1],
               s_omega_i[(3 * threadIdx.x) + 2]);
  const Vec3 V(s_omega_o[3 * threadIdx.x], s_omega_o[(3 * threadIdx.x) + 1],
               s_omega_o[(3 * threadIdx.x) + 2]);
  const Vec3 n(s_n[3 * threadIdx.x], s_n[(3 * threadIdx.x) + 1],
               s_n[(3 * threadIdx.x) + 2]);
  const float P_c = P_c_[threadIdx.x];
  const float P_cg = P_cg_[threadIdx.x];

  // compute brdf
  const Vec3 H = (V + L).normalize();

  // Multiply 1 vector - each color channel gets same derivative
  const float dBRDF_dP_cg =
      0.25F * P_c * G_r(L, V, n) * F_r(L, H) * dD_r_dP_cg(H, P_cg, n);
  // write out result
  result[(3 * global_id)] = dBRDF_dP_cg;
  result[(3 * global_id) + 1] = dBRDF_dP_cg;
  result[(3 * global_id) + 2] = dBRDF_dP_cg;
}

extern "C" void principled_brdf_backward_P_cg_cuda_impl(
    const float *__restrict__ omega_i, const float *__restrict__ omega_o,
    const float *__restrict__ P_c, const float *__restrict__ P_cg,
    const float *__restrict__ n, float *__restrict__ result, size_t N) {

  const size_t N_BLOCKS = (N + N_THREADS - 1) / N_THREADS;

  // shared memory for coalesced read in of float3s
  const size_t shared_memory_bytes = sizeof(float) * 9 * N;
  principled_brdf_backward_P_cg_kernel<<<N_BLOCKS, N_THREADS,
                                        shared_memory_bytes>>>(
      omega_i, omega_o, P_c, P_cg, n, result, N);
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
