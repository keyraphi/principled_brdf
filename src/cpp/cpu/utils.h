#pragma once
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <cmath> 

namespace nb = nanobind;

// Forward declarations
using FlexScalarCPU = nb::ndarray<const float, nb::shape<-1>, nb::c_contig, nb::device::cpu>;
using FlexVec3CPU = nb::ndarray<const float, nb::shape<-1, 3>, nb::c_contig, nb::device::cpu>;
using ScalarArrayCPU = nb::ndarray<const float, nb::shape<-1>, nb::c_contig, nb::device::cpu>;
using Vec3ArrayCPU = nb::ndarray<const float, nb::shape<-1, 3>, nb::c_contig, nb::device::cpu>;

namespace cpu {

// Create default arrays of size [N]
Vec3ArrayCPU create_default_vec3(size_t N, float x, float y, float z);
ScalarArrayCPU create_default_scalar(size_t N, float value);

// Broadcast arrays from [1] to [N] or [1, 3] to [N, 3]
ScalarArrayCPU broadcast_scalar(const FlexScalarCPU& source, size_t N);
Vec3ArrayCPU broadcast_vec3(const FlexVec3CPU& source, size_t N);


struct Vec3 {
    float x, y, z;

    // Constructor from individual components
    Vec3(float x_, float y_, float z_) : x{x_}, y{y_}, z{z_} {}

    // Default constructor (initializes to zero)
    Vec3() = default;

    // Vector addition
    Vec3 operator+(const Vec3& rhs) const {
        return Vec3(x + rhs.x, y + rhs.y, z + rhs.z);
    }

    // Scalar multiplication (both directions)
    Vec3 operator*(float scalar) const {
        return Vec3(x * scalar, y * scalar, z * scalar);
    }
    friend Vec3 operator*(float scalar, const Vec3& vec) {
        return vec * scalar;
    }

    // Dot product (vector multiplication)
    float operator*(const Vec3& rhs) const {
        return x * rhs.x + y * rhs.y + z * rhs.z;
    }

    // Scalar division
    Vec3 operator/(float scalar) const {
        return Vec3(x / scalar, y / scalar, z / scalar);
    }

    // In-place normalization
    Vec3& normalize() {
        float len = std::sqrt(x*x + y*y + z*z);
        if (len > 0) {
            x /= len;
            y /= len;
            z /= len;
        }
        return *this;
    }

    // Element-wise Hadamard product
    Vec3 hadamard(const Vec3& rhs) const {
        return Vec3(x * rhs.x, y * rhs.y, z * rhs.z);
    }
};

} // namespace cpu
