#pragma once

// CPU implementation headers will go here

// For now, just declare the functions we'll implement
void disney_brdf_forward_cpu(const float* input, float* output, int64_t size);
void disney_brdf_backward_cpu(const float* input, const float* grad_output, 
                             float* grad_input, int64_t size);
