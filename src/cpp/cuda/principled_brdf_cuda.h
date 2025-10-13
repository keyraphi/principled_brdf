#pragma once
#include <cstddef>

extern "C" {

void cuda_dummy_add(const float* a, const float* b, float* result, size_t n);

void principled_brdf_cuda_forward(const float *omega_i, const float *omega_o,
                                  const float *P_b, const float *P_m,
                                  const float *P_ss, const float *P_s,
                                  const float *P_r, const float *P_st,
                                  const float *P_ani, const float *P_sh,
                                  const float *P_sht, const float *P_c,
                                  const float *P_cg, const float *n,
                                  float *result, size_t N);
}
