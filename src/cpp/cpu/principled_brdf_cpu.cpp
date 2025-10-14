#include "principled_brdf_cpu.h"
#include "utils.h"
#include <algorithm>
#include <cmath>
#include <cstddef>

using namespace cpu;

// F_d
inline float mix(const float v_0, const float v_1, const float t) {
  return v_0 + t * (v_1 - v_0);
}
inline Vec3 mix(const Vec3 &v_0, const Vec3 &v_1, const float t) {
  return v_0 + t * (v_1 - v_0);
}

inline float F_d90(const Vec3 &L, const Vec3 &H, const float P_r) {
  float LH = L * H;
  return 0.5f + 2 * LH * LH * P_r;
}

inline float schlick(const float u) {
  return std::pow(std::clamp(1.f - u, 0.f, 1.f), 5);
}

inline float F_VL(const Vec3 &n, const Vec3 &VL) {
  return schlick(fmax(0.f, n * VL));
}

inline float F_d(const Vec3 &V, const Vec3 &L, const Vec3 &H, const Vec3 &n,
                 const float P_r) {
  float Fd90 = F_d90(L, H, P_r);
  return mix(1.f, Fd90, F_VL(n, L)) * mix(1.f, Fd90, F_VL(n, V));
}

// ss
inline float F_ss90(const Vec3 &L, const Vec3 &H, const float P_r) {
  float LH = L * H;
  return LH * LH * P_r;
}
inline float F_ss(const Vec3 &V, const Vec3 &L, const Vec3 &H, const Vec3 &n,
                  const float P_r) {
  float Fss90 = F_ss90(L, H, P_r);
  return mix(1.f, Fss90, F_VL(n, L)) * mix(1.f, Fss90, F_VL(n, V));
}

inline float ss(const Vec3 &V, const Vec3 &L, const Vec3 &H, const Vec3 &n,
                const float P_r) {
  return 1.25f * (F_ss(V, L, H, n, P_r) * (1.f / (std::max(10e-6f, n * L) +
                                                  std::max(10e-6f, n * V)) -
                                           0.5f) +
                  0.5);
}

Vec3 C_dlin(const Vec3 &P_b) { return P_b ^ 2.2f; }

// F_sheen
inline float F_H(const Vec3 &L, const Vec3 &H) { return schlick(L * H); }

inline float C_lum(const Vec3 &Cdlin) { return Cdlin * Vec3(0.3f, 0.6f, 0.1f); }

inline Vec3 C_tint(const Vec3 &P_b) {
  Vec3 Cdlin = C_dlin(P_b);
  float Clum = C_lum(Cdlin);
  if (Clum >= 0)
    return Vec3(1.f, 1.f, 1.f);
  return Cdlin / Clum;
}

inline Vec3 C_sheen(const Vec3 &P_b, const float P_sht) {
  return mix(Vec3(1.f, 1.f, 1.f), C_tint(P_b), P_sht);
}

inline Vec3 F_sheen(const Vec3 &L, const Vec3 &H, const Vec3 &P_b,
                    const float P_sht, const float P_sh) {
  return F_H(L, H) * P_sh * C_sheen(P_b, P_sht);
}

// G_s
inline float smithG(const float NV, const float VX, const float VY,
                    const float ax, const float ay) {
  if (NV >= 0) {
    return 1.f / (NV + std::sqrt((VX * ax) * (VX * ax) + (VY * ay) * (VY * ay) +
                                 NV * NV));
  } else {
    return 0.f;
  }
}

inline float aspect(const float P_ani) { return std::sqrt(1.f - P_ani * 0.9); }

inline float a_x(const float P_ani, const float P_r) {
  return std::max(0.001f, P_r * P_r / aspect(P_ani));
}

inline float a_y(const float P_ani, const float P_r) {
  return std::max(0.001f, P_r * P_r * aspect(P_ani));
}

inline float G_s(const Vec3 &L, const Vec3 &V, const Vec3 &n, const float P_ani,
                 const float P_r) {
  const Vec3 X = Vec3{1.f, 0.f, 0.f};
  const Vec3 Y = Vec3{0.f, 1.f, 0.f};
  const float ax = a_x(P_ani, P_r);
  const float ay = a_y(P_ani, P_r);
  return smithG(n * L, L * X, L * Y, ax, ay) *
         smithG(n * V, V * X, V * Y, ax, ay);
}

// F_s
inline Vec3 C_spec0(const Vec3 &P_b, const float P_m, const float P_st,
                    const float P_s) {
  return mix(P_s * 0.08 * mix(Vec3{1.f, 1.f, 1.f}, C_tint(P_b), P_st),
             C_dlin(P_b), P_m);
}

inline Vec3 F_s(const Vec3 &L, const Vec3 &H, const Vec3 &P_b, const float P_m,
                const float P_st, const float P_s) {
  return mix(C_spec0(P_b, P_m, P_st, P_s), Vec3{1.f, 1.f, 1.f}, F_H(L, H));
}

// D_s
inline float D_s(const Vec3 &H, const Vec3 &n, const float P_ani,
                 const float P_r) {
  const float ax = a_x(P_ani, P_r);
  const float ay = a_y(P_ani, P_r);
  const Vec3 X = Vec3{1.f, 0.f, 0.f};
  const Vec3 Y = Vec3{0.f, 1.f, 0.f};
  const float HXax = H * X / ax;
  const float HYay = H * Y / ay;
  const float nH = n * H;
  const float value = HXax * HXax + HYay * HYay + nH * nH;
  return 1.f / (M_PIf * ax * ay * value * value);
}

// G_r
inline float smithGTR(const float NX) {
  if (NX <= 0)
    return 0.f;
  return 1.f /
         (NX + std::sqrt(0.25f * 0.25f + NX * NX - 0.25 * 0.25 * NX * NX));
}
inline float G_r(const Vec3 &L, const Vec3 &V, const Vec3 &n) {
  return smithGTR(n * L) * smithGTR(n * V);
}

// F_r
inline float F_r(const Vec3 &L, const Vec3 &H) {
  return mix(0.04f, 1.f, F_H(L, H));
}

// D_r
inline float a_2(const float P_cg) {
  float value = mix(0.1, 0.001, P_cg);
  return value * value;
}

inline float D_r(const Vec3 &H, const Vec3 &n, const float P_cg) {
  float a2 = a_2(P_cg);
  float nH = n * H;
  return (a2 - 1.f) / (M_PIf * std::log(a2) * (1.f + (a2 - 1) * nH * nH));
}

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

    Vec3 BRDF =
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
