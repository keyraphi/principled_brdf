#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "cpu/principled_brdf_cpu.h"
#include "cuda/principled_brdf_cuda.h"

namespace nb = nanobind;

// Forward declarations for CUDA functions
extern "C" {
void *cuda_allocate(size_t n);
void cuda_free(void *ptr);
void cuda_dummy_add(const float *a, const float *b, float *result, size_t n);
}

// Dummy function that dispatches to CPU or CUDA implementation
nb::ndarray<float> dummy_add(const nb::ndarray<float> &a,
                             const nb::ndarray<float> &b) {
  if (a.shape(0) != b.shape(0) || a.shape(1) != b.shape(1)) {
    throw std::runtime_error("Shape mismatch");
  }

  size_t n = a.shape(0) * a.shape(1);
  size_t shape[2] = {a.shape(0), a.shape(1)};

  if (a.device_type() == nb::device::cpu::value) {
    // Allocate CPU memory
    float *result_data = new float[n];

    // CPU implementation
    const float *a_data = a.data();
    const float *b_data = b.data();

    for (size_t i = 0; i < n; ++i) {
      result_data[i] = a_data[i] + b_data[i];
    }

    // Create ndarray that takes ownership of the memory
    nb::capsule owner(result_data,
                      [](void *p) noexcept { delete[] (float *)p; });

    return nb::ndarray<float>(result_data, 2, shape, owner);
  } else if (a.device_type() == nb::device::cuda::value) {
    // Allocate CUDA memory using our CUDA function
    float *result_data = static_cast<float *>(cuda_allocate(n * sizeof(float)));
    if (!result_data) {
      throw std::runtime_error("Failed to allocate CUDA memory");
    }

    // Call CUDA implementation
    const float *a_data = a.data();
    const float *b_data = b.data();
    cuda_dummy_add(a_data, b_data, result_data, n);

    // Create ndarray that takes ownership of the CUDA memory
    nb::capsule owner(result_data, [](void *p) noexcept { cuda_free(p); });

    return nb::ndarray<float>(result_data, 2, shape, owner,
                              nullptr, // strides (nullptr = contiguous)
                              nb::dtype<float>(), nb::device::cuda::value,
                              a.device_id());
  } else {
    throw std::runtime_error("Unsupported device type");
  }
}

NB_MODULE(principled_brdf_functions, m) {
  m.doc() = "Principled BRDF with partial derivatives";

  m.def("dummy_add", &dummy_add, "Add two arrays");
}
