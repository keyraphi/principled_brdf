#include "principled_brdf_cpu.h"
#include "../common/common_utils.h"
#include <cstddef>
#include <math.h>

void principled_brdf_cpu_forward(
    const float *__restrict__ omega_i, const float *__restrict__ omega_o,
    const float *__restrict__ P_b, const float *__restrict__ P_m,
    const float *__restrict__ P_ss, const float *__restrict__ P_s,
    const float *__restrict__ P_r, const float *__restrict__ P_st,
    const float *__restrict__ P_ani, const float *__restrict__ P_sh,
    const float *__restrict__ P_sht, const float *__restrict__ P_c,
    const float *__restrict__ P_cg, const float *__restrict__ n,
    float *__restrict__ result, size_t N) {
#pragma omp parallel for
  for (size_t i = 0; i < N; ++i) {
    const Vec3 L(omega_i[i * 3], omega_i[i * 3 + 1], omega_i[i * 3 + 2]);
    const Vec3 V(omega_o[i * 3], omega_o[i * 3 + 1], omega_o[i * 3 + 2]);
    const Vec3 H = (V + L).normalize();

    Vec3 _n{n[i * 3], n[i * 3 + 1], n[i * 3 + 2]};
    Vec3 _P_b{P_b[i * 3], P_b[i * 3 + 1], P_b[i * 3 + 2]};

    const Vec3 BRDF =
        (M_1_PIf *
             mix(F_d(V, L, H, _n, P_r[i]), ss(V, L, H, _n, P_r[i]), P_ss[i]) *
             C_dlin(_P_b) +
         F_sheen(L, H, _P_b, P_sht[i], P_sh[i])) *
            (1.f - P_m[i]) +
        G_s(L, V, _n, P_ani[i], P_r[i]) *
            F_s(L, H, _P_b, P_m[i], P_st[i], P_s[i]) *
            D_s(H, _n, P_ani[i], P_r[i]) +
        0.25f * P_c[i] * G_r(L, V, _n) * F_r(L, H) * D_r(H, _n, P_cg[i]);

    result[i * 3] = BRDF.x;
    result[i * 3 + 1] = BRDF.y;
    result[i * 3 + 2] = BRDF.z;
  }
}
