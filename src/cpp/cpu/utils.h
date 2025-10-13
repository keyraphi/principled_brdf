#pragma once
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

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

} // namespace cpu
