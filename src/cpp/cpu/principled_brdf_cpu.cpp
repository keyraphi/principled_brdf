#include "principled_brdf_cpu.h"
#include "utils.h"
#include <cmath>
#include <cstddef>

using namespace cpu;

void principled_brdf_cpu_forward(const float *omega_i, const float *omega_o,
                                 const float *P_b, const float *P_m,
                                 const float *P_ss, const float *P_s,
                                 const float *P_r, const float *P_st,
                                 const float *P_ani, const float *P_sh,
                                 const float *P_sht, const float *P_c,
                                 const float *P_cg, const float *n,
                                 float *result, size_t N) {
#pragma omp parallel for
  for (size_t i = 0; i < N; ++i) {
    const Vec3 L(omega_i[i * 3], omega_i[i * 3 + 1], omega_i[i * 3 + 2]);
    const Vec3 V(omega_o[i * 3], omega_o[i * 3 + 1], omega_o[i * 3 + 2]);
    const Vec3 H = (V + L).normalize();

    Vec3 _n{n[i * 3], n[i * 3 + 1], n[i * 3 + 2]};
    Vec3 _P_b{P_b[i * 3], P_b[i * 3 + 1], P_b[i * 3 + 2]};

    // TODO There is also a dependencie on L, V and H most of the time.
    Vec3 BRDF =
        (M_1_PIf * mix(F_d(_n, P_r[i]), ss(_n, P_r[i]), P_ss[i]) * C_dlin(_P_b) +
         F_sheen(_P_b, P_sht[i], P_sh[i]) *
            (1 - P_m[i]) +
        G_s(_n, P_ani[i], P_r[i]) * F_s(_P_b, P_m[i], P_st[i], P_s[i]) * D_s(_n, P_ani[i], P_r[i]) + 0.25f * P_c[i] * G_r(_n) * F_r(L, H) * D_r(_n, P_cg[i]);
  }
}
