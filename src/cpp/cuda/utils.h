#pragma once
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

namespace nb = nanobind;

namespace cuda {
// Forward declarations
using FlexScalarCUDA = nb::ndarray<const float, nb::shape<-1>, nb::c_contig, nb::device::cuda>;
using FlexVec3CUDA = nb::ndarray<const float, nb::shape<-1, 3>, nb::c_contig, nb::device::cuda>;
using ScalarArrayCUDA = nb::ndarray<const float, nb::shape<-1>, nb::c_contig, nb::device::cuda>;
using Vec3ArrayCUDA = nb::ndarray<const float, nb::shape<-1, 3>, nb::c_contig, nb::device::cuda>;

// CUDA memory management functions
extern "C" {
    void* cuda_allocate(size_t n);
    void cuda_free(void* ptr);
    void cuda_broadcast_scalar(float* dst, float value, size_t N);
    void cuda_broadcast_vec3(float* dst, const float* src, size_t N);
}


// Create default arrays of size [N]
Vec3ArrayCUDA create_default_vec3(size_t N, float x, float y, float z);
ScalarArrayCUDA create_default_scalar(size_t N, float value);

// Broadcast arrays from [1] to [N] or [1, 3] to [N, 3]
ScalarArrayCUDA broadcast_scalar(const FlexScalarCUDA& source, size_t N);
Vec3ArrayCUDA broadcast_vec3(const FlexVec3CUDA& source, size_t N);

} // namespace cuda
