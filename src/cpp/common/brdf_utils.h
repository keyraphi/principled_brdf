#pragma once
#include <cmath>
#include <cuda_runtime_api.h>

#ifdef __CUDACC__
#define HOST_DEVICE __host__ __device__
#else
#define HOST_DEVICE
#endif

struct Mat3x3;

struct Vec3 {
  float x, y, z;

  // Constructor from individual components
  HOST_DEVICE Vec3(float x_, float y_, float z_) : x{x_}, y{y_}, z{z_} {}

  // Default constructor (initializes to zero)
  HOST_DEVICE Vec3() = default;

  // Vector addition
  HOST_DEVICE Vec3 operator+(const Vec3 &rhs) const {
    return {x + rhs.x, y + rhs.y, z + rhs.z};
  }
  HOST_DEVICE Vec3 operator+(const float rhs) const {
    return {x + rhs, y + rhs, z + rhs};
  }
  HOST_DEVICE Vec3 operator-(const Vec3 &rhs) const {
    return {x - rhs.x, y - rhs.y, z - rhs.z};
  }
  HOST_DEVICE Vec3 operator-() const { return {-x, -y, -z}; }

  // Scalar multiplication (both directions)
  HOST_DEVICE Vec3 operator*(float scalar) const {
    return {x * scalar, y * scalar, z * scalar};
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
    return {x / scalar, y / scalar, z / scalar};
  }

  HOST_DEVICE Vec3 operator^(float exponent) const {
    return {powf(x, exponent), powf(y, exponent), powf(z, exponent)};
  }

  // In-place normalization
  HOST_DEVICE Vec3 &normalize() {
    float len = sqrtf(x * x + y * y + z * z);
    if (len > 0) {
      float inv_len = 1.F / len;
      x *= inv_len;
      y *= inv_len;
      z *= inv_len;
    }
    return *this;
  }

  // outer product (resulting in Mat3x3)
  HOST_DEVICE Mat3x3 odot(const Vec3 &rhs) const;
};

struct Mat3x3 {
  float m[9]; // row major

  HOST_DEVICE Mat3x3() = default;
  HOST_DEVICE Mat3x3(const Mat3x3 &other) = default;
  HOST_DEVICE Mat3x3 &operator=(const Mat3x3 &other) = default;
  HOST_DEVICE Mat3x3(Mat3x3 &&other) = default;
  HOST_DEVICE Mat3x3 &operator=(Mat3x3 &&other) = default;

  HOST_DEVICE Mat3x3(float constant) {
    // we trust the compiler to unroll automatically
    for (int i = 0; i < 9; ++i) {
      m[i] = constant;
    }
  }
  static HOST_DEVICE auto diag(const Vec3 &vec) -> Mat3x3 {
    Mat3x3 result{0.F};
    result.m[0] = vec.x;
    result.m[4] = vec.y;
    result.m[8] = vec.z;
    return result;
  }

  // Mat3x3 * Vec3 (Matrix times 3x1 Vector)
  HOST_DEVICE inline Vec3 operator*(const Vec3 &vec) const {
    Vec3 result;

    result.x = m[0] * vec.x + m[1] * vec.y + m[2] * vec.z;
    result.y = m[3] * vec.x + m[4] * vec.y + m[5] * vec.z;
    result.z = m[6] * vec.x + m[7] * vec.y + m[8] * vec.z;

    return result;
  }
  // Vec3 * Mat3x3 (1x3 Vector times Matrix)
  HOST_DEVICE friend Vec3 operator*(const Vec3 &vec, const Mat3x3 &mat) {
    Vec3 result;

    result.x = vec.x * mat.m[0] + vec.y * mat.m[3] + vec.z * mat.m[6];
    result.y = vec.x * mat.m[1] + vec.y * mat.m[4] + vec.z * mat.m[7];
    result.z = vec.x * mat.m[2] + vec.y * mat.m[5] + vec.z * mat.m[8];

    return result;
  }
  // Mat3x3 * Mat3x3 (Matrix Multiplication)
  HOST_DEVICE Mat3x3 operator*(const Mat3x3 &other) const {
    Mat3x3 result{0.F}; // Initialize result to zero

    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        float sum = 0.F;
        for (int k = 0; k < 3; ++k) {
          sum += this->m[i * 3 + k] * other.m[k * 3 + j];
        }
        result.m[i * 3 + j] = sum;
      }
    }
    return result;
  }

  // Mat3x3 + Mat3x3 (Element-wise addition)
  HOST_DEVICE Mat3x3 operator+(const Mat3x3 &other) const {
    Mat3x3 result{0.F};
    for (int i = 0; i < 9; ++i) {
      result.m[i] = this->m[i] + other.m[i];
    }
    return result;
  }

  // Mat3x3 - Mat3x3 (Element-wise subtraction)
  HOST_DEVICE Mat3x3 operator-(const Mat3x3 &other) const {
    Mat3x3 result{0.F};
    for (int i = 0; i < 9; ++i) {
      result.m[i] = this->m[i] - other.m[i];
    }
    return result;
  }

  // Mat3x3 * float (Element-wise multiplication)
  HOST_DEVICE Mat3x3 operator*(float scalar) const {
    Mat3x3 result{*this};
    for (int i = 0; i < 9; ++i) {
      result.m[i] *= scalar;
    }
    return result;
  }

  // Mat3x3 + float (Element-wise addition)
  HOST_DEVICE Mat3x3 operator+(float scalar) const {
    Mat3x3 result{*this};
    for (int i = 0; i < 9; ++i) {
      result.m[i] += scalar;
    }
    return result;
  }

  // Mat3x3 - float (Element-wise subtraction)
  HOST_DEVICE Mat3x3 operator-(float scalar) const {
    Mat3x3 result{*this};
    for (int i = 0; i < 9; ++i) {
      result.m[i] -= scalar;
    }
    return result;
  }

  // Mat3x3 / float (Element-wise division)
  HOST_DEVICE Mat3x3 operator/(float scalar) const {
    float inv_scalar = 1.F / scalar;
    return (*this) * inv_scalar;
  }

  // Global float * Mat3x3
  HOST_DEVICE friend Mat3x3 operator*(float scalar, const Mat3x3 &matrix) {
    return matrix * scalar; // Reuses the member function
  }
  // Global float + Mat3x3
  HOST_DEVICE friend Mat3x3 operator+(float scalar, const Mat3x3 &matrix) {
    return matrix + scalar; // Reuses the member function
  }

  // Global float - Mat3x3
  HOST_DEVICE friend Mat3x3 operator-(float scalar, const Mat3x3 &matrix) {
    Mat3x3 result{0.F};
    for (int i = 0; i < 9; ++i) {
      result.m[i] = scalar - matrix.m[i];
    }
    return result;
  }
};

// outer product of two Vec3
HOST_DEVICE inline Mat3x3 Vec3::odot(const Vec3 &rhs) const {
  Mat3x3 result;
  result.m[0] = x * rhs.x;
  result.m[1] = x * rhs.y;
  result.m[2] = x * rhs.z;
  result.m[3] = y * rhs.x;
  result.m[4] = y * rhs.y;
  result.m[5] = y * rhs.z;
  result.m[6] = z * rhs.x;
  result.m[7] = z * rhs.y;
  result.m[8] = z * rhs.z;
  return result;
}

// F_d
template <typename T>
HOST_DEVICE inline T mix(const T &v_0, const T &v_1, const float t) {
  return v_0 + t * (v_1 - v_0);
}

HOST_DEVICE inline float F_d90(const Vec3 &L, const Vec3 &H, const float P_r) {
  float LH = L * H;
  return 0.5F + 2.F * LH * LH * P_r;
}

HOST_DEVICE inline float schlick(const float u) {
  float value = u < 0.F ? 0.F : (u > 1.F ? 1.F : u); // clamp to [0, 1]
  return value * value * value * value * value;      // value^5
}

HOST_DEVICE inline float F_VL(const Vec3 &n, const Vec3 &VL) {
  return schlick(fmax(0.F, n * VL));
}

HOST_DEVICE inline float F_d(const Vec3 &V, const Vec3 &L, const Vec3 &H,
                             const Vec3 &n, const float P_r) {
  float Fd90 = F_d90(L, H, P_r);
  return mix(1.F, Fd90, F_VL(n, L)) * mix(1.F, Fd90, F_VL(n, V));
}

// ss
HOST_DEVICE inline float F_ss90(const Vec3 &L, const Vec3 &H, const float P_r) {
  float LH = L * H;
  return LH * LH * P_r;
}
HOST_DEVICE inline float F_ss(const Vec3 &V, const Vec3 &L, const Vec3 &H,
                              const Vec3 &n, const float P_r) {
  float Fss90 = F_ss90(L, H, P_r);
  return mix(1.F, Fss90, F_VL(n, L)) * mix(1.F, Fss90, F_VL(n, V));
}

HOST_DEVICE inline float ss(const Vec3 &V, const Vec3 &L, const Vec3 &H,
                            const Vec3 &n, const float P_r) {
  return 1.25F *
         (F_ss(V, L, H, n, P_r) *
              (1.F / (fmaxf(10e-6F, n * L) + fmaxf(10e-6F, n * V)) - 0.5F) +
          0.5F);
}

HOST_DEVICE inline Vec3 C_dlin(const Vec3 &P_b) { return P_b ^ 2.2F; }

// F_sheen
HOST_DEVICE inline float F_H(const Vec3 &L, const Vec3 &H) {
  return schlick(L * H);
}

HOST_DEVICE inline float C_lum(const Vec3 &Cdlin) {
  return Cdlin * Vec3(0.3F, 0.6F, 0.1F);
}

HOST_DEVICE inline Vec3 C_tint(const Vec3 &P_b) {
  Vec3 Cdlin = C_dlin(P_b);
  float Clum = C_lum(Cdlin);
  if (Clum >= 0) {
    return {1.F, 1.F, 1.F};
  }
  return Cdlin / Clum;
}

HOST_DEVICE inline Vec3 C_sheen(const Vec3 &P_b, const float P_sht) {
  return mix(Vec3(1.F, 1.F, 1.F), C_tint(P_b), P_sht);
}

HOST_DEVICE inline Vec3 F_sheen(const Vec3 &L, const Vec3 &H, const Vec3 &P_b,
                                const float P_sht, const float P_sh) {
  return F_H(L, H) * P_sh * C_sheen(P_b, P_sht);
}

// G_s
HOST_DEVICE inline float smithG(const float NV, const float VX, const float VY,
                                const float ax, const float ay) {
  if (NV >= 0) {
    return 1.F / (NV + sqrtf((VX * ax) * (VX * ax) + (VY * ay) * (VY * ay) +
                             NV * NV));
  } else {
    return 0.F;
  }
}

HOST_DEVICE inline float aspect(const float P_ani) {
  return sqrtf(1.F - P_ani * 0.9);
}

HOST_DEVICE inline float a_x(const float P_ani, const float P_r) {
  return fmaxf(0.001F, P_r * P_r / aspect(P_ani));
}

HOST_DEVICE inline float a_y(const float P_ani, const float P_r) {
  return fmaxf(0.001F, P_r * P_r * aspect(P_ani));
}

HOST_DEVICE inline float G_s(const Vec3 &L, const Vec3 &V, const Vec3 &n,
                             const float P_ani, const float P_r) {
  const Vec3 X = Vec3{1.F, 0.F, 0.F};
  const Vec3 Y = Vec3{0.F, 1.F, 0.F};
  const float ax = a_x(P_ani, P_r);
  const float ay = a_y(P_ani, P_r);
  return smithG(n * L, L * X, L * Y, ax, ay) *
         smithG(n * V, V * X, V * Y, ax, ay);
}

// F_s
HOST_DEVICE inline Vec3 C_spec0(const Vec3 &P_b, const float P_m,
                                const float P_st, const float P_s) {
  return mix(P_s * 0.08F * mix(Vec3{1.F, 1.F, 1.F}, C_tint(P_b), P_st),
             C_dlin(P_b), P_m);
}

HOST_DEVICE inline Vec3 F_s(const Vec3 &L, const Vec3 &H, const Vec3 &P_b,
                            const float P_m, const float P_st,
                            const float P_s) {
  return mix(C_spec0(P_b, P_m, P_st, P_s), Vec3{1.F, 1.F, 1.F}, F_H(L, H));
}

// D_s
HOST_DEVICE inline float D_s(const Vec3 &H, const Vec3 &n, const float P_ani,
                             const float P_r) {
  const float ax = a_x(P_ani, P_r);
  const float ay = a_y(P_ani, P_r);
  const Vec3 X = Vec3{1.F, 0.F, 0.F};
  const Vec3 Y = Vec3{0.F, 1.F, 0.F};
  const float HXax = H * X / ax;
  const float HYay = H * Y / ay;
  const float nH = n * H;
  const float value = HXax * HXax + HYay * HYay + nH * nH;
  return 1.F / (M_PIf * ax * ay * value * value);
}

// G_r
HOST_DEVICE inline float smithGTR(const float NX) {
  if (NX <= 0)
    return 0.F;
  return 1.F / (NX + sqrtf(0.25f * 0.25f + NX * NX - 0.25 * 0.25 * NX * NX));
}
HOST_DEVICE inline float G_r(const Vec3 &L, const Vec3 &V, const Vec3 &n) {
  return smithGTR(n * L) * smithGTR(n * V);
}

// F_r
HOST_DEVICE inline float F_r(const Vec3 &L, const Vec3 &H) {
  return mix(0.04f, 1.F, F_H(L, H));
}

// D_r
HOST_DEVICE inline float a_2(const float P_cg) {
  float value = mix(0.1F, 0.001F, P_cg);
  return value * value;
}

HOST_DEVICE inline float D_r(const Vec3 &H, const Vec3 &n, const float P_cg) {
  float a2 = a_2(P_cg);
  float nH = n * H;
  return (a2 - 1.F) / (M_PIf * logf(a2) * (1.F + (a2 - 1) * nH * nH));
}

// dBRDF_dP_b
// //////////////////////////////////////////////////////////////////////
HOST_DEVICE inline Mat3x3 dC_dlin_dP_b(const Vec3 &P_b) {
  return 2.2F * Mat3x3::diag(P_b ^ 1.2F);
}

// Note, result is a row vector!
HOST_DEVICE inline Vec3 dC_lum_dP_b(const Vec3 &P_b) {
  return 2.2F * Vec3(0.3F, 0.6F, 0.1F) *
         Mat3x3::diag(P_b ^ 1.2F); // vector matrix multiplication from left:
                                   // 1x3 vector * 3x3 matrix yields 1x3 vector
}

HOST_DEVICE inline Mat3x3 dC_tint_dP_b(const Vec3 &P_b, const float P_sht) {
  const Vec3 cdlin = C_dlin(P_b);
  const float clum = C_lum(cdlin);
  if (clum <= 0.F) {
    return Mat3x3(0.F);
  }
  // odot: multiplication of 3x1 vector with 1x3 vector resulting in a Mat3x3
  return P_sht * (dC_dlin_dP_b(P_b) * clum - cdlin.odot(dC_lum_dP_b(P_b))) /
         (clum * clum);
}

HOST_DEVICE inline Mat3x3 dC_sheen_dP_b(const Vec3 &P_b, const float P_sht) {
  return P_sht * dC_tint_dP_b(P_b, P_sht);
}

HOST_DEVICE inline Mat3x3 dF_sheen_dP_b(const Vec3 &L, const Vec3 &H,
                                        const Vec3 &P_b, const float P_sh,
                                        const float P_sht) {
  return F_H(L, H) * P_sh * dC_sheen_dP_b(P_b, P_sht);
}

HOST_DEVICE inline Mat3x3 dC_spec0_dP_b(const Vec3 &P_b, const float P_s,
                                        const float P_st, const float P_m) {
  return (P_s * 0.08F * P_st * dC_tint_dP_b(P_b, P_st)) * (1.F - P_m) +
         P_m * dC_dlin_dP_b(P_b);
}

HOST_DEVICE inline Mat3x3 dF_s_dP_b(const Vec3 &L, const Vec3 &H,
                                    const Vec3 &P_b, const float P_s,
                                    const float P_st, const float P_m) {
  return (1.F - F_H(L, H)) * dC_spec0_dP_b(P_b, P_s, P_st, P_m);
}

// dBRDF_dP_m ////////////////////////////////////////////////////////////////
HOST_DEVICE inline Vec3 dC_spec0_dP_m(const Vec3 &P_b, const float P_s,
                                      const float P_st) {
  return C_dlin(P_b) -
         P_s * 0.08F *
             (Vec3{1.F, 1.F, 1.F} + P_st * (C_tint(P_b) - Vec3{1.F, 1.F, 1.F}));
}
HOST_DEVICE inline Vec3 dF_s_dP_m(const Vec3 &L, const Vec3 &H, const Vec3 &P_b,
                                  const float P_st, const float P_s) {
  return (1.F - F_H(L, H)) * dC_spec0_dP_m(P_b, P_s, P_st);
}

// dBRDF_dP_s ////////////////////////////////////////////////////////////////
HOST_DEVICE inline Vec3 dC_spec0_dP_s(const Vec3 &P_b, const float P_m,
                                      const float P_st) {
  return 0.08F * mix(Vec3{1.F, 1.F, 1.F}, C_tint(P_b), P_st) * (1.F - P_m);
}

// dBRDF_dP_r ////////////////////////////////////////////////////////////////
HOST_DEVICE inline float dF_d90_dP_r(const Vec3 &L, const Vec3 &H) {
  return 2.F * (L * H) * (L * H);
}
HOST_DEVICE inline float dF_d_dP_r(const Vec3 &L, const Vec3 &V, const Vec3 &H,
                                   const float P_r, const Vec3 &n) {
  return dF_d90_dP_r(L, H) *
         (2.F * (F_d90(L, H, P_r) - 1.F) * F_VL(n, L) * F_VL(n, V) +
          F_VL(n, L) + F_VL(n, V));
}

HOST_DEVICE inline float dF_ss90_dP_r(const Vec3 &L, const Vec3 &H) {
  return (L * H) * (L * H);
}
HOST_DEVICE inline float dF_ss_dP_r(const Vec3 &L, const Vec3 &V, const Vec3 &H,
                                    const float P_r, const Vec3 &n) {
  return dF_ss90_dP_r(L, H) *
         (2.F * (F_ss90(L, H, P_r) - 1.F) * F_VL(n, L) * F_VL(n, V) +
          F_VL(n, L) + F_VL(n, V));
}
HOST_DEVICE inline float dss_dP_r(const Vec3 &L, const Vec3 &V, const Vec3 &H,
                                  const float P_r, const Vec3 &n) {
  return 1.25F * (1.F / (fmaxf(1e-6F, n * L) + fmaxf(1e-10F, n * V)) - 0.5F) *
         dF_ss_dP_r(L, V, H, P_r, n);
}

HOST_DEVICE inline float da_x_dP_r(const float P_r, const float P_ani) {
  if (P_r * P_r / aspect(P_ani) < 0.01F) {
    return 0.F;
  }
  return 2.F * P_r / aspect(P_ani);
}
HOST_DEVICE inline float da_y_dP_r(const float P_r, const float P_ani) {
  if (P_r * P_r * aspect(P_ani) < 0.01F) {
    return 0.F;
  }
  return 2 * P_r * aspect(P_ani);
}
HOST_DEVICE inline float dHXax_dP_r(const Vec3 &H, const Vec3 &X,
                                    const float P_r, const float P_ani,
                                    const float ax) {
  return -(H * X) * da_x_dP_r(P_r, P_ani) / (ax * ax);
}
HOST_DEVICE inline float dHYay_dP_r(const Vec3 &H, const Vec3 &Y,
                                    const float P_r, const float P_ani,
                                    const float ay) {
  return -(H * Y) * da_y_dP_r(P_r, P_ani) / (ay * ay);
}
HOST_DEVICE inline float dHXax2_dP_r(const Vec3 &H, const Vec3 &X,
                                     const float P_r, const float P_ani,
                                     const float HXax, const float ax) {
  return 2.F * HXax * dHXax_dP_r(H, X, P_r, P_ani, ax);
}
HOST_DEVICE inline float dHYay2_dP_r(const Vec3 &H, const Vec3 &Y,
                                     const float P_r, const float P_ani,
                                     const float HYay, const float ay) {
  return 2.F * HYay * dHYay_dP_r(H, Y, P_r, P_ani, ay);
}
HOST_DEVICE inline float dHXax2HYay2nH2_dP_r(const Vec3 &H, const Vec3 &X,
                                             const Vec3 &Y, const float P_r,
                                             const float P_ani,
                                             const float HXax, const float HYay,
                                             const float ax, const float ay) {
  return dHXax2_dP_r(H, X, P_r, P_ani, HXax, ax) +
         dHYay2_dP_r(H, Y, P_r, P_ani, HYay, ay);
}
HOST_DEVICE inline float
dHXax2HYay2nH2_2_dP_r(const Vec3 &H, const Vec3 &X, const Vec3 &Y,
                      const float P_r, const float P_ani,
                      const float HXax2HYay2nH2, const float HXax,
                      const float HYay, const float ax, const float ay) {
  return 2.F * (HXax2HYay2nH2 *
                dHXax2HYay2nH2_dP_r(H, X, Y, P_r, P_ani, HXax, HYay, ax, ay));
}

HOST_DEVICE inline float
dnominator_dD_s_dP_r(const Vec3 &H, const Vec3 &X, const Vec3 &Y,
                     const float P_r, const float P_ani,
                     const float HXax2HYay2nH2, const float HXax,
                     const float HYay, const float ax, const float ay) {
  return M_PIf * (ay * (HXax2HYay2nH2 * HXax2HYay2nH2 * da_x_dP_r(P_r, P_ani) +
                        ax * dHXax2HYay2nH2_2_dP_r(H, X, Y, P_r, P_ani,
                                                   HXax2HYay2nH2, HXax, HYay,
                                                   ax, ay))) +
         ax * (HXax2HYay2nH2 * HXax2HYay2nH2 * da_y_dP_r(P_r, P_ani));
}
HOST_DEVICE inline float dD_s_dP_r(const Vec3 &H, const float P_r,
                                   const float P_ani, const Vec3 &n) {
  const float ax = a_x(P_ani, P_r);
  const float ay = a_y(P_ani, P_r);
  const Vec3 X = Vec3{1.F, 0.F, 0.F};
  const Vec3 Y = Vec3{0.F, 1.F, 0.F};
  const float HXax = H * X / ax;
  const float HYay = H * Y / ay;
  const float nH = n * H;
  const float HXax2HYay2nH2 = (HXax * HXax + HYay * HYay + nH * nH);
  const float denom = (M_PIf * a_x(P_ani, P_r) * a_y(P_ani, P_r) *
                       HXax2HYay2nH2 * HXax2HYay2nH2);
  return dnominator_dD_s_dP_r(H, X, Y, P_r, P_ani, HXax2HYay2nH2, HXax, HYay,
                              ax, ay) /
         (denom * denom);
}

HOST_DEVICE inline float dS_LV_dP_r(const float P_r, const float P_ani,
                                    const float LVXax, const float LVYay,
                                    const float nLV, const float LVX,
                                    const float LVY) {
  const float S_LV = sqrtf(LVXax * LVXax + LVYay * LVYay + nLV * nLV);
  return (1.F / S_LV) * (LVXax * LVX * da_x_dP_r(P_r, P_ani) +
                         (LVYay * LVY) * LVY * da_y_dP_r(P_r, P_ani));
}
HOST_DEVICE inline float dsmithG_LV_dP_r(const float P_r, const float P_ani,
                                         const float nLV, const float LVX,
                                         const float LVY, const float ax,
                                         const float ay) {
  if (nLV < 0) {
    return 0.F;
  }
  const float smithg = smithG(nLV, LVX, LVY, ax, ay);
  return smithg * smithg *
         dS_LV_dP_r(P_r, P_ani, LVX * ax, LVY * ay, nLV, LVX, LVY);
}
HOST_DEVICE inline float dG_s_dP_r(const Vec3 &L, const Vec3 &V,
                                   const float P_r, const float P_ani,
                                   const Vec3 &n) {
  const float ax = a_x(P_ani, P_r);
  const float ay = a_y(P_ani, P_r);
  const Vec3 X = Vec3{1.F, 0.F, 0.F};
  const Vec3 Y = Vec3{0.F, 1.F, 0.F};

  return smithG(n * L, L * X, L * Y, ax, ay) *
             dsmithG_LV_dP_r(P_r, P_ani, n * V, V * X, V * Y, ax, ay) +
         smithG(n * n, V * X, V * Y, ax, ay) *
             dsmithG_LV_dP_r(P_r, P_ani, n * L, L * X, L * Y, ax, ay);
}

// dBRDF_dP_st ////////////////////////////////////////////////////////////////
HOST_DEVICE inline Vec3 dC_spec0_dP_st(const Vec3 &P_b, const float P_m,
                                       const float P_s) {
  return P_s * 0.08F * (C_tint(P_b) - Vec3({1.F, 1.F, 1.F})) * (1.F - P_m);
}
HOST_DEVICE inline Vec3 dF_s_dP_st(const Vec3 &L, const Vec3 &H,
                                   const Vec3 &P_b, const float P_m,
                                   const float P_s) {
  return (1.F - F_H(L, H)) * dC_spec0_dP_st(P_b, P_m, P_s);
}

// dBRDF_dP_ani ////////////////////////////////////////////////////////////////
HOST_DEVICE inline float daspect_dP_ani(const float P_ani) {
  return -1.45F / sqrtf(1.F - 0.9F * P_ani);
}

HOST_DEVICE inline float da_x_daspect(const float P_r, const float P_ani) {
  const float value = P_r * P_r / aspect(P_ani);
  if (value > 0.001) {
    return -value / aspect(P_ani);
  }
  return 0.F;
}

HOST_DEVICE inline float da_y_daspect(const float P_r) {
  const float value = P_r * P_r;
  if (value > 0.001) {
    return value;
  }
  return 0.F;
}

HOST_DEVICE inline float dsmithG_LV_da_y(const float ND, const float DX,
                                         const float DY, const float ax,
                                         const float ay) {
  const float S =
      sqrtf((DX * ax) * (DX * ax) + (DY * ay) * (DY * ay) + (ND * ND));
  const float dS_da_y = 2.F * DY * DY * ay / (2 * S);
  return -dS_da_y / ((ND + S) * (ND + S));
}
HOST_DEVICE inline float dsmithG_LV_da_x(const float ND, const float DX,
                                         const float DY, const float ax,
                                         const float ay) {
  const float S =
      sqrtf((DX * ax) * (DX * ax) + (DY * ay) * (DY * ay) + (ND * ND));
  const float dS_da_x = 2.F * DX * DX * ax / (2 * S);
  return -dS_da_x / ((ND + S) * (ND + S));
}
HOST_DEVICE inline float dG_s_da_x(const Vec3 &L, const Vec3 &V,
                                   const float P_ani, const float P_r,
                                   const Vec3 &n) {
  const Vec3 X = Vec3{1.F, 0.F, 0.F};
  const Vec3 Y = Vec3{0.F, 1.F, 0.F};
  const float ax = a_x(P_ani, P_r);
  const float ay = a_y(P_ani, P_r);
  const float G1 = smithG(n * L, L * X, L * Y, ax, ay);
  const float G2 = smithG(n * V, V * X, V * Y, ax, ay);
  return G2 * dsmithG_LV_da_x(n * L, L * X, L * Y, ax, ay) +
         G1 * dsmithG_LV_da_x(n * V, V * X, V * Y, ax, ay);
}
HOST_DEVICE inline float dG_s_da_y(const Vec3 &L, const Vec3 &V,
                                   const float P_ani, const float P_r,
                                   const Vec3 &n) {
  const Vec3 X = Vec3{1.F, 0.F, 0.F};
  const Vec3 Y = Vec3{0.F, 1.F, 0.F};
  const float ax = a_x(P_ani, P_r);
  const float ay = a_y(P_ani, P_r);
  const float G1 = smithG(n * L, L * X, L * Y, ax, ay);
  const float G2 = smithG(n * V, V * X, V * Y, ax, ay);
  return G2 * dsmithG_LV_da_y(n * L, L * X, L * Y, ax, ay) +
         G1 * dsmithG_LV_da_y(n * V, V * X, V * Y, ax, ay);
}
HOST_DEVICE inline float dD_s_da_x(const Vec3 &H, const float P_r,
                                   const float P_ani, const Vec3 &n) {
  const Vec3 X = Vec3{1.F, 0.F, 0.F};
  const Vec3 Y = Vec3{0.F, 1.F, 0.F};
  const float ax = a_x(P_ani, P_r);
  const float ay = a_y(P_ani, P_r);
  const float E = (H * X / ax) * (H * X / ax) + (H * Y / ay) * (H * Y / ay) +
                  (n * H) * (n * H);
  return -D_s(H, n, P_ani, P_r) *
         (1.F / ax - (4.F * (H * X) * (H * X)) / (E * ax * ax * ax));
}
HOST_DEVICE inline float dD_s_da_y(const Vec3 &H, const float P_r,
                                   const float P_ani, const Vec3 &n) {
  const Vec3 X = Vec3{1.F, 0.F, 0.F};
  const Vec3 Y = Vec3{0.F, 1.F, 0.F};
  const float ax = a_x(P_ani, P_r);
  const float ay = a_y(P_ani, P_r);
  const float E = (H * X / ax) * (H * X / ax) + (H * Y / ay) * (H * Y / ay) +
                  (n * H) * (n * H);
  return -D_s(H, n, P_ani, P_r) *
         (1.F / ay - (4.F * (H * Y) * (H * Y)) / (E * ay * ay * ay));
}
HOST_DEVICE inline Vec3 dBRDF_da_x(const Vec3 &L, const Vec3 &V, const Vec3 &H,
                                   const Vec3 &P_b, const float P_m,
                                   const float P_st, const float P_s,
                                   const float P_ani, const float P_r,
                                   const Vec3 &n) {
  return F_s(L, H, P_b, P_m, P_st, P_s) *
         (D_s(H, n, P_ani, P_r) * dG_s_da_x(L, V, P_ani, P_r, n) +
          G_s(L, V, n, P_ani, P_r) * dD_s_da_x(H, P_r, P_ani, n));
}

HOST_DEVICE inline Vec3 dBRDF_da_y(const Vec3 &L, const Vec3 &V, const Vec3 &H,
                                   const Vec3 &P_b, const float P_m,
                                   const float P_st, const float P_s,
                                   const float P_ani, const float P_r,
                                   const Vec3 &n) {
  return F_s(L, H, P_b, P_m, P_st, P_s) *
         (D_s(H, n, P_ani, P_r) * dG_s_da_y(L, V, P_ani, P_r, n) +
          G_s(L, V, n, P_ani, P_r) * dD_s_da_y(H, P_r, P_ani, n));
}

// dBRDF_dP_sh ////////////////////////////////////////////////////////////////
HOST_DEVICE inline Vec3 dF_sheen_dP_sh(const Vec3 &L, const Vec3 &H,
                                       const Vec3 &P_b, const float P_sht) {
  return F_H(L, H) * C_sheen(P_b, P_sht);
}

// dBRDF_dP_sht ////////////////////////////////////////////////////////////////
HOST_DEVICE inline Vec3 dC_sheen_dP_sht(const Vec3 &P_b){
  return C_tint(P_b) - Vec3{1.F, 1.F, 1.F};
}
HOST_DEVICE inline Vec3 dF_sheen_dP_sht(const Vec3 &L, const Vec3 &H,
                                       const Vec3 &P_b, const float P_sh) {
  return F_H(L, H) * P_sh * dC_sheen_dP_sht(P_b);
}
