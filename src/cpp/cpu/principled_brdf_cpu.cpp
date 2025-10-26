#include "principled_brdf_cpu.h"
#include "../common/brdf_utils.h"
#include <cstddef>
#include <math.h>

void principled_brdf_forward_cpu_impl(
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
    const Vec3 L(omega_i[i * 3], omega_i[(i * 3) + 1], omega_i[(i * 3) + 2]);
    const Vec3 V(omega_o[i * 3], omega_o[(i * 3) + 1], omega_o[(i * 3) + 2]);
    const Vec3 H = (V + L).normalize();

    Vec3 n_{n[i * 3], n[(i * 3) + 1], n[(i * 3) + 2]};
    Vec3 P_b_{P_b[i * 3], P_b[(i * 3) + 1], P_b[(i * 3) + 2]};

    const Vec3 BRDF =
        (M_1_PIf *
             mix(F_d(V, L, H, n_, P_r[i]), ss(V, L, H, n_, P_r[i]), P_ss[i]) *
             C_dlin(P_b_) +
         F_sheen(L, H, P_b_, P_sht[i], P_sh[i])) *
            (1.F - P_m[i]) +
        G_s(L, V, n_, P_ani[i], P_r[i]) *
            F_s(L, H, P_b_, P_m[i], P_st[i], P_s[i]) *
            D_s(H, n_, P_ani[i], P_r[i]) +
        0.25F * P_c[i] * G_r(L, V, n_) * F_r(L, H) * D_r(H, n_, P_cg[i]);

    result[i * 3] = BRDF.x;
    result[(i * 3) + 1] = BRDF.y;
    result[(i * 3) + 2] = BRDF.z;
  }
}

void principled_brdf_backward_P_b_cpu_impl(
    const float *__restrict__ omega_i, const float *__restrict__ omega_o,
    const float *__restrict__ P_b, const float *__restrict__ P_m,
    const float *__restrict__ P_ss, const float *__restrict__ P_s,
    const float *__restrict__ P_r, const float *__restrict__ P_st,
    const float *__restrict__ P_ani, const float *__restrict__ P_sh,
    const float *__restrict__ P_sht, const float *__restrict__ n,
    float *__restrict__ result, size_t N) {
#pragma omp parallel for
  for (size_t i = 0; i < N; ++i) {
    const Vec3 L(omega_i[i * 3], omega_i[(i * 3) + 1], omega_i[(i * 3) + 2]);
    const Vec3 V(omega_o[i * 3], omega_o[(i * 3) + 1], omega_o[(i * 3) + 2]);
    const Vec3 H = (V + L).normalize();

    Vec3 n_{n[i * 3], n[(i * 3) + 1], n[(i * 3) + 2]};
    Vec3 P_b_{P_b[i * 3], P_b[(i * 3) + 1], P_b[(i * 3) + 2]};

    const Mat3x3 dBRDFdP_b =
        ((1.F - P_m[i]) * M_1_PIf *
         mix(F_d(V, L, H, n_, P_r[i]), ss(V, L, H, n_, P_r[i]), P_ss[i]) *
         dC_dlin_dP_b(P_b_)) +
        dF_sheen_dP_b(L, H, P_b_, P_sh[i], P_sht[i]) +
        (D_s(H, n_, P_ani[i], P_r[i]) * G_s(L, V, n_, P_ani[i], P_r[i]) *
         dF_s_dP_b(L, H, P_b_, P_s[i], P_st[i], P_m[i]));

    result[9 * i + 0] = dBRDFdP_b.m[0];
    result[9 * i + 1] = dBRDFdP_b.m[1];
    result[9 * i + 2] = dBRDFdP_b.m[2];
    result[9 * i + 3] = dBRDFdP_b.m[3];
    result[9 * i + 4] = dBRDFdP_b.m[4];
    result[9 * i + 5] = dBRDFdP_b.m[5];
    result[9 * i + 6] = dBRDFdP_b.m[6];
    result[9 * i + 7] = dBRDFdP_b.m[7];
    result[9 * i + 8] = dBRDFdP_b.m[8];
  }
}

void principled_brdf_backward_P_m_cpu_impl(
    const float *__restrict__ omega_i, const float *__restrict__ omega_o,
    const float *__restrict__ P_b, const float *__restrict__ P_ss,
    const float *__restrict__ P_s, const float *__restrict__ P_r,
    const float *__restrict__ P_st, const float *__restrict__ P_ani,
    const float *__restrict__ P_sh, const float *__restrict__ P_sht,
    const float *__restrict__ n, float *__restrict__ result, size_t N) {
#pragma omp parallel for
  for (size_t i = 0; i < N; ++i) {
    const Vec3 L(omega_i[i * 3], omega_i[(i * 3) + 1], omega_i[(i * 3) + 2]);
    const Vec3 V(omega_o[i * 3], omega_o[(i * 3) + 1], omega_o[(i * 3) + 2]);
    const Vec3 H = (V + L).normalize();

    Vec3 n_{n[i * 3], n[(i * 3) + 1], n[(i * 3) + 2]};
    Vec3 P_b_{P_b[i * 3], P_b[(i * 3) + 1], P_b[(i * 3) + 2]};

    const Vec3 dBRDF_dP_m =
        -(M_1_PIf *
              mix(F_d(V, L, H, n_, P_r[i]), ss(V, L, H, n_, P_r[i]), P_ss[i]) *
              C_dlin(P_b_) +
          F_sheen(L, H, P_b_, P_sht[i], P_sh[i])) +
        G_s(L, V, n_, P_ani[i], P_r[i]) * D_s(H, n_, P_ani[i], P_r[i]) *
            dF_s_dP_m(L, H, P_b_, P_st[i], P_s[i]);

    result[i * 3] = dBRDF_dP_m.x;
    result[(i * 3) + 1] = dBRDF_dP_m.y;
    result[(i * 3) + 2] = dBRDF_dP_m.z;
  }
}

void principled_brdf_backward_P_ss_cpu_impl(
    const float *__restrict__ omega_i, const float *__restrict__ omega_o,
    const float *__restrict__ P_b, const float *__restrict__ P_m,
    const float *__restrict__ P_r, const float *__restrict__ n,
    float *__restrict__ result, size_t N) {
#pragma omp parallel for
  for (size_t i = 0; i < N; ++i) {
    const Vec3 L(omega_i[i * 3], omega_i[(i * 3) + 1], omega_i[(i * 3) + 2]);
    const Vec3 V(omega_o[i * 3], omega_o[(i * 3) + 1], omega_o[(i * 3) + 2]);
    const Vec3 H = (V + L).normalize();

    Vec3 n_{n[i * 3], n[(i * 3) + 1], n[(i * 3) + 2]};
    Vec3 P_b_{P_b[i * 3], P_b[(i * 3) + 1], P_b[(i * 3) + 2]};

    const Vec3 dBRDF_dP_ss =
        (1.F - P_m[i]) * M_1_PIf * C_dlin(P_b_) *
        (ss(V, L, H, n_, P_r[i]) - F_d(V, L, H, n_, P_r[i]));

    result[i * 3] = dBRDF_dP_ss.x;
    result[(i * 3) + 1] = dBRDF_dP_ss.y;
    result[(i * 3) + 2] = dBRDF_dP_ss.z;
  }
}

void principled_brdf_backward_P_s_cpu_impl(
    const float *__restrict__ omega_i, const float *__restrict__ omega_o,
    const float *__restrict__ P_b, const float *__restrict__ P_m,
    const float *__restrict__ P_st, const float *__restrict__ n,
    float *__restrict__ result, size_t N) {
#pragma omp parallel for
  for (size_t i = 0; i < N; ++i) {
    const Vec3 L(omega_i[i * 3], omega_i[(i * 3) + 1], omega_i[(i * 3) + 2]);
    const Vec3 V(omega_o[i * 3], omega_o[(i * 3) + 1], omega_o[(i * 3) + 2]);
    const Vec3 H = (V + L).normalize();

    Vec3 n_{n[i * 3], n[(i * 3) + 1], n[(i * 3) + 2]};
    Vec3 P_b_{P_b[i * 3], P_b[(i * 3) + 1], P_b[(i * 3) + 2]};

    const Vec3 dBRDF_dP_s =
        (1.F - F_H(L, H)) * dC_spec0_dP_s(P_b_, P_m[i], P_st[i]);

    result[i * 3] = dBRDF_dP_s.x;
    result[(i * 3) + 1] = dBRDF_dP_s.y;
    result[(i * 3) + 2] = dBRDF_dP_s.z;
  }
}

void principled_brdf_backward_P_r_cpu_impl(
    const float *__restrict__ omega_i, const float *__restrict__ omega_o,
    const float *__restrict__ P_b, const float *__restrict__ P_m,
    const float *__restrict__ P_ss, const float *__restrict__ P_s,
    const float *__restrict__ P_r, const float *__restrict__ P_st,
    const float *__restrict__ P_ani, const float *__restrict__ n,
    float *__restrict__ result, size_t N) {
#pragma omp parallel for
  for (size_t i = 0; i < N; ++i) {
    const Vec3 L(omega_i[i * 3], omega_i[(i * 3) + 1], omega_i[(i * 3) + 2]);
    const Vec3 V(omega_o[i * 3], omega_o[(i * 3) + 1], omega_o[(i * 3) + 2]);
    const Vec3 H = (V + L).normalize();

    Vec3 n_{n[i * 3], n[(i * 3) + 1], n[(i * 3) + 2]};
    Vec3 P_b_{P_b[i * 3], P_b[(i * 3) + 1], P_b[(i * 3) + 2]};

    const Vec3 dBRDF_dP_r =
        (1.F - P_m[i]) * M_1_PIf * C_dlin(P_b_) *
            ((1.F - P_ss[i]) * dF_d_dP_r(L, V, H, P_r[i], n_) +
             dss_dP_r(L, V, H, P_r[i], n_)) +
        F_s(L, H, P_b_, P_m[i], P_st[i], P_s[i]) *
            (dG_s_dP_r(L, V, P_r[i], P_ani[i], n_) *
                 D_s(H, n_, P_ani[i], P_r[i]) +
             G_s(L, V, n_, P_ani[i], P_r[i]) *
                 dD_s_dP_r(H, P_r[i], P_ani[i], n_));

    result[i * 3] = dBRDF_dP_r.x;
    result[(i * 3) + 1] = dBRDF_dP_r.y;
    result[(i * 3) + 2] = dBRDF_dP_r.z;
  }
}

void principled_brdf_backward_P_st_cpu_impl(
    const float *__restrict__ omega_i, const float *__restrict__ omega_o,
    const float *__restrict__ P_b, const float *__restrict__ P_m,
    const float *__restrict__ P_s, const float *__restrict__ P_r,
    const float *__restrict__ P_ani, const float *__restrict__ n,
    float *__restrict__ result, size_t N) {
#pragma omp parallel for
  for (size_t i = 0; i < N; ++i) {
    const Vec3 L(omega_i[i * 3], omega_i[(i * 3) + 1], omega_i[(i * 3) + 2]);
    const Vec3 V(omega_o[i * 3], omega_o[(i * 3) + 1], omega_o[(i * 3) + 2]);
    const Vec3 H = (V + L).normalize();

    Vec3 n_{n[i * 3], n[(i * 3) + 1], n[(i * 3) + 2]};
    Vec3 P_b_{P_b[i * 3], P_b[(i * 3) + 1], P_b[(i * 3) + 2]};

    const Vec3 dBRDF_dP_st = G_s(L, V, n_, P_ani[i], P_r[i]) *
                             D_s(H, n_, P_ani[i], P_r[i]) *
                             dF_s_dP_st(L, H, P_b_, P_m[i], P_s[i]);

    result[i * 3] = dBRDF_dP_st.x;
    result[(i * 3) + 1] = dBRDF_dP_st.y;
    result[(i * 3) + 2] = dBRDF_dP_st.z;
  }
}
