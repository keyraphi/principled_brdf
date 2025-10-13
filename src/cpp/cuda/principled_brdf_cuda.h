#pragma once

#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif

// CUDA memory management
void* cuda_allocate(size_t size);
void cuda_free(void* ptr);

// CUDA operations
void cuda_dummy_add(const float* a, const float* b, float* result, size_t n);

#ifdef __cplusplus
}
#endif

