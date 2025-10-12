#include "disney_brdf_cpu.h"

void disney_brdf_forward_cpu(const float* input, float* output, int64_t size) {
    // TODO: Implement Disney BRDF forward pass
    for (int64_t i = 0; i < size; ++i) {
        output[i] = input[i]; // Placeholder
    }
}

void disney_brdf_backward_cpu(const float* input, const float* grad_output, 
                             float* grad_input, int64_t size) {
    // TODO: Implement Disney BRDF backward pass
    for (int64_t i = 0; i < size; ++i) {
        grad_input[i] = grad_output[i]; // Placeholder
    }
}
