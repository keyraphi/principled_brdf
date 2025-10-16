#pragma once
#include <algorithm>
#include <cmath>
#include <cuda_runtime_api.h>

#ifdef __CUDACC__
#define HOST_DEVICE __host__ __device__
#else
#define HOST_DEVICE
#endif

struct Vec3 {
  float x, y, z;

  // Constructor from individual components
  HOST_DEVICE Vec3(float x_, float y_, float z_) : x{x_}, y{y_}, z{z_} {}

  // Default constructor (initializes to zero)
  HOST_DEVICE Vec3() = default;

  // Vector addition
  HOST_DEVICE Vec3 operator+(const Vec3 &rhs) const {
    return Vec3(x + rhs.x, y + rhs.y, z + rhs.z);
  }
  HOST_DEVICE Vec3 operator+(const float rhs) const {
    return Vec3(x + rhs, y + rhs, z + rhs);
  }
  HOST_DEVICE Vec3 operator-(const Vec3 &rhs) const {
    return Vec3(x - rhs.x, y - rhs.y, z - rhs.z);
  }

  // Scalar multiplication (both directions)
  HOST_DEVICE Vec3 operator*(float scalar) const {
    return Vec3(x * scalar, y * scalar, z * scalar);
  }
  HOST_DEVICE friend Vec3 operator*(float scalar, const Vec3 &vec) {
    return vec * scalar;
  }

  // Dot product (vector multiplication)
  HOST_DEVICE float operator*(const Vec3 &rhs) const {
    return x * rhs.x + y * rhs.y + z * rhs.z;
  }

  // Scalar division
  HOST_DEVICE Vec3 operator/(float scalar) const {
    return Vec3(x / scalar, y / scalar, z / scalar);
  }

  HOST_DEVICE Vec3 operator^(float exponent) const {
    return Vec3(powf(x, exponent), powf(y, exponent), powf(z, exponent));
  }

  // In-place normalization
  HOST_DEVICE Vec3 &normalize() {
    float len = sqrtf(x * x + y * y + z * z);
    if (len > 0) {
      float inv_len = 1.f / len;
      x *= inv_len;
      y *= inv_len;
      z *= inv_len;
    }
    return *this;
  }

  // Element-wise Hadamard product
  HOST_DEVICE Vec3 hadamard(const Vec3 &rhs) const {
    return Vec3(x * rhs.x, y * rhs.y, z * rhs.z);
  }
};

// F_d
HOST_DEVICE inline float mix(const float v_0, const float v_1, const float t) {
  return v_0 + t * (v_1 - v_0);
}
HOST_DEVICE inline Vec3 mix(const Vec3 &v_0, const Vec3 &v_1, const float t) {
  return v_0 + t * (v_1 - v_0);
}

HOST_DEVICE inline float F_d90(const Vec3 &L, const Vec3 &H, const float P_r) {
  float LH = L * H;
  return 0.5f + 2 * LH * LH * P_r;
}

HOST_DEVICE inline float schlick(const float u) {
  float value = u < 0.f ? 0.f : (u > 1.f ? 1.f: u); // clamp to [0, 1]
  return value * value * value * value * value; // value^5
}

HOST_DEVICE inline float F_VL(const Vec3 &n, const Vec3 &VL) {
  return schlick(fmax(0.f, n * VL));
}

HOST_DEVICE inline float F_d(const Vec3 &V, const Vec3 &L, const Vec3 &H,
                             const Vec3 &n, const float P_r) {
  float Fd90 = F_d90(L, H, P_r);
  return mix(1.f, Fd90, F_VL(n, L)) * mix(1.f, Fd90, F_VL(n, V));
}

// ss
HOST_DEVICE inline float F_ss90(const Vec3 &L, const Vec3 &H, const float P_r) {
  float LH = L * H;
  return LH * LH * P_r;
}
HOST_DEVICE inline float F_ss(const Vec3 &V, const Vec3 &L, const Vec3 &H,
                              const Vec3 &n, const float P_r) {
  float Fss90 = F_ss90(L, H, P_r);
  return mix(1.f, Fss90, F_VL(n, L)) * mix(1.f, Fss90, F_VL(n, V));
}

HOST_DEVICE inline float ss(const Vec3 &V, const Vec3 &L, const Vec3 &H,
                            const Vec3 &n, const float P_r) {
  return 1.25f * (F_ss(V, L, H, n, P_r) * (1.f / (std::max(10e-6f, n * L) +
                                                  std::max(10e-6f, n * V)) -
                                           0.5f) +
                  0.5);
}

HOST_DEVICE inline Vec3 C_dlin(const Vec3 &P_b) { return P_b ^ 2.2f; }

// F_sheen
HOST_DEVICE inline float F_H(const Vec3 &L, const Vec3 &H) {
  return schlick(L * H);
}

HOST_DEVICE inline float C_lum(const Vec3 &Cdlin) {
  return Cdlin * Vec3(0.3f, 0.6f, 0.1f);
}

HOST_DEVICE inline Vec3 C_tint(const Vec3 &P_b) {
  Vec3 Cdlin = C_dlin(P_b);
  float Clum = C_lum(Cdlin);
  if (Clum >= 0)
    return Vec3(1.f, 1.f, 1.f);
  return Cdlin / Clum;
}

HOST_DEVICE inline Vec3 C_sheen(const Vec3 &P_b, const float P_sht) {
  return mix(Vec3(1.f, 1.f, 1.f), C_tint(P_b), P_sht);
}

HOST_DEVICE inline Vec3 F_sheen(const Vec3 &L, const Vec3 &H, const Vec3 &P_b,
                                const float P_sht, const float P_sh) {
  return F_H(L, H) * P_sh * C_sheen(P_b, P_sht);
}

// G_s
HOST_DEVICE inline float smithG(const float NV, const float VX, const float VY,
                                const float ax, const float ay) {
  if (NV >= 0) {
    return 1.f / (NV + std::sqrt((VX * ax) * (VX * ax) + (VY * ay) * (VY * ay) +
                                 NV * NV));
  } else {
    return 0.f;
  }
}

HOST_DEVICE inline float aspect(const float P_ani) {
  return std::sqrt(1.f - P_ani * 0.9);
}

HOST_DEVICE inline float a_x(const float P_ani, const float P_r) {
  return std::max(0.001f, P_r * P_r / aspect(P_ani));
}

HOST_DEVICE inline float a_y(const float P_ani, const float P_r) {
  return std::max(0.001f, P_r * P_r * aspect(P_ani));
}

HOST_DEVICE inline float G_s(const Vec3 &L, const Vec3 &V, const Vec3 &n,
                             const float P_ani, const float P_r) {
  const Vec3 X = Vec3{1.f, 0.f, 0.f};
  const Vec3 Y = Vec3{0.f, 1.f, 0.f};
  const float ax = a_x(P_ani, P_r);
  const float ay = a_y(P_ani, P_r);
  return smithG(n * L, L * X, L * Y, ax, ay) *
         smithG(n * V, V * X, V * Y, ax, ay);
}

// F_s
HOST_DEVICE inline Vec3 C_spec0(const Vec3 &P_b, const float P_m,
                                const float P_st, const float P_s) {
  return mix(P_s * 0.08 * mix(Vec3{1.f, 1.f, 1.f}, C_tint(P_b), P_st),
             C_dlin(P_b), P_m);
}

HOST_DEVICE inline Vec3 F_s(const Vec3 &L, const Vec3 &H, const Vec3 &P_b,
                            const float P_m, const float P_st,
                            const float P_s) {
  return mix(C_spec0(P_b, P_m, P_st, P_s), Vec3{1.f, 1.f, 1.f}, F_H(L, H));
}

// D_s
HOST_DEVICE inline float D_s(const Vec3 &H, const Vec3 &n, const float P_ani,
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
HOST_DEVICE inline float smithGTR(const float NX) {
  if (NX <= 0)
    return 0.f;
  return 1.f /
         (NX + std::sqrt(0.25f * 0.25f + NX * NX - 0.25 * 0.25 * NX * NX));
}
HOST_DEVICE inline float G_r(const Vec3 &L, const Vec3 &V, const Vec3 &n) {
  return smithGTR(n * L) * smithGTR(n * V);
}

// F_r
HOST_DEVICE inline float F_r(const Vec3 &L, const Vec3 &H) {
  return mix(0.04f, 1.f, F_H(L, H));
}

// D_r
HOST_DEVICE inline float a_2(const float P_cg) {
  float value = mix(0.1, 0.001, P_cg);
  return value * value;
}

HOST_DEVICE inline float D_r(const Vec3 &H, const Vec3 &n, const float P_cg) {
  float a2 = a_2(P_cg);
  float nH = n * H;
  return (a2 - 1.f) / (M_PIf * std::log(a2) * (1.f + (a2 - 1) * nH * nH));
}
