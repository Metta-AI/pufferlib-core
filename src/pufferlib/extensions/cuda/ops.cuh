#pragma once

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cmath>

// These functions are ported from pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/
// PyTorch defines these kernels anonymously and with extra saftey and features
// This is a lower-overhead port of the math for use in fused kernels
// The caller handles templating, so these can just be simple C-style code
#ifndef CUDART_INF_F
#define CUDART_INF_F __int_as_float(0x7f800000)
#endif

#define SOFTPLUS_BETA 1.0f
#define SOFTPLUS_THRESHOLD 20.0f

__device__ __forceinline__ float softplus_fwd(float x) {
    float x_scaled = x * SOFTPLUS_BETA;
    if (x_scaled > SOFTPLUS_THRESHOLD) {
        return x;
    } else {
        return log1pf(expf(x_scaled)) / SOFTPLUS_BETA;
    }
}

__device__ __forceinline__ float softplus_bwd(float grad_output, float x) {
    float beta_x = SOFTPLUS_BETA * x;
    if (beta_x > SOFTPLUS_THRESHOLD) {
        return grad_output;
    } else {
        float exp_beta_x = expf(beta_x);
        return grad_output * (exp_beta_x / (1.0f + exp_beta_x));
    }
}

__device__ __forceinline__ float relu(float x) {
    return fmaxf(0.0f, x);
}

__device__ __forceinline__ float relu_backward(float x, float grad_output) {
    return (x > 0.0f) ? grad_output : 0.0f;
}

__device__ __forceinline__ float sigmoid(float x) {
    float z = expf(-fabsf(x));
    return x >= 0.0f ? 1.0f / (1.0f + z) : z / (1.0f + z);
}

__device__ __forceinline__ float sigmoid_backward(float x, float grad_output) {
    float sig = sigmoid(x);
    return grad_output * sig * (1.0f - sig);
}

__device__ __inline__ float fast_tanh(float x) {
  const float plus_9 = 9.0f;
  const float minus_9 = -9.0f;
  float v1 = fminf(x, plus_9);
  v1 = fmaxf(v1, minus_9);

  const float alpha_1 = 4.89352455891786e-03f;
  const float alpha_3 = 6.37261928875436e-04f;
  const float alpha_5 = 1.48572235717979e-05f;
  const float alpha_7 = 5.12229709037114e-08f;
  const float alpha_9 = -8.60467152213735e-11f;
  const float alpha_11 = 2.00018790482477e-13f;
  const float alpha_13 = -2.76076847742355e-16f;
  const float beta_0 = 4.89352518554385e-03f;
  const float beta_2 = 2.26843463243900e-03f;
  const float beta_4 = 1.18534705686654e-04f;
  const float beta_6 = 1.19825839466702e-06f;

  // Horner's method. Matches PyTorch implementation
  float v2 = v1 * v1;
  float p = v2 * alpha_13 + alpha_11;
  p = v2 * p + alpha_9;
  p = v2 * p + alpha_7;
  p = v2 * p + alpha_5;
  p = v2 * p + alpha_3;
  p = v2 * p + alpha_1;
  p = v1 * p;

  float q = v2 * beta_6 + beta_4;
  q = v2 * q + beta_2;
  q = v2 * q + beta_0;

  return p / q;
}

__device__ __inline__ float fast_sigmoid(float x) {
  const float one_v = 1.0f;
  const float half_v = 0.5f;
  const float zero_v = 0.0f;
  float x2 = x * half_v;
  float y = fast_tanh(x2);
  float z = (y + one_v) * half_v;
  return fminf(one_v, fmaxf(zero_v, z));
}

__device__ __forceinline__ float tilde_relu_fwd(float x) {
    if (x >= 0.0f) {
        return x + 0.5f;
    } else {
        return fast_sigmoid(x);
    }
}

__device__ __forceinline__ float tilde_relu_bwd(float x, float grad_output) {
    if (x >= 0.0f) {
        return grad_output * 1.0f;
    } else {
        float sig = fast_sigmoid(x);
        return grad_output * sig * (1.0f - sig);
    }
}

__device__ __forceinline__ float lerp(float a, float b, float w) {
    float diff = b - a;
    if (fabsf(w) < 0.5f) {
        return a + w * diff;
    } else {
        return b - diff * (1.0f - w);
    }
}

__device__ __forceinline__ float lerp_backward_a(float a, float b, float w, float grad_output) {
    return grad_output * (1.0f - w);
}

__device__ __forceinline__ float lerp_backward_b(float a, float b, float w, float grad_output) {
    return grad_output * w;
}

__device__ __forceinline__ float lerp_backward_w(float a, float b, float w, float grad_output) {
    float diff = b - a;
    if (fabsf(w) < 0.5f) {
        return grad_output * diff;
    } else {
        return grad_output * (-diff) * -1.0f; // derivative of (1 - w)
    }
}

__device__ __forceinline__ float mingru_gate(float h) {
    return h >= 0.0f ? h + 0.5f : sigmoid(h);
}

__device__ __forceinline__ float mingru_gate_backward(float h, float grad_output) {
    if (h > 0.0f) {
        return grad_output * 1.0f;
    } else {
        float sig = sigmoid(h);
        return grad_output * sig * (1.0f - sig);
    }
}

__device__ __forceinline__ float logaddexp(float a, float b) {
    float m = fmaxf(a, b);
    float v = fminf(a, b);
    float diff = v - m;
    return (diff < -88.0f) ? m : m + log1pf(expf(diff));
}

__device__ __forceinline__ float exp_safe(float x) {
    return (x > 88.0f) ? 1.651e38f : ((x < -88.0f) ? 0.0f : expf(x));
}

__device__ __forceinline__ void cumsum_forward(const float* x, float* y, int T) {
    float sum = 0.0f;
    for (int t = 0; t < T; t++) {
        sum += x[t];
        y[t] = sum;
    }
}

__device__ __forceinline__ void cumsum_backward(const float* grad_y, float* grad_x, int T) {
    float running = 0.0f;
    for (int t = T - 1; t >= 0; t--) {
        running += grad_y[t];
        grad_x[t] = running;
    }
}

__device__ __forceinline__ void logcumsumexp_forward(const float* x, float* y, int T) {
    float lse = -CUDART_INF_F;
    for (int t = 0; t < T; t++) {
        lse = (lse == -CUDART_INF_F) ? x[t] : logaddexp(lse, x[t]);
        y[t] = lse;
    }
}

__device__ __forceinline__ void logcumsumexp_backward(
    const float* x,       // input: x[s]
    const float* y,       // output: y[t]
    const float* grad_y,  // dL/dy[t]
    float* grad_x,        // output: dL/dx[s]
    int T
) {
    float running = 0.0f;
    for (int t = T - 1; t >= 0; t--) {
        float weight = exp_safe(x[t] - y[t]);
        running += grad_y[t] * weight;
        grad_x[t] = running;
    }
}

// ============================================================================
// log_coeffs_and_values: Single-element ops for fused scan
// These are ported directly from log_coeffs_and_values_kernel in kernels.cu
// ============================================================================

// Forward: computes log_coeffs and log_values for mingru scan
// log_coeffs = -softplus(gate)
// log_values = -softplus(-gate) + log_tilde_h
//   where log_tilde_h = log(relu(hidden) + 0.5) if hidden >= 0
//                     = -softplus(-hidden)      if hidden < 0
__device__ __forceinline__ void log_coeffs_and_values_fwd(
    float gate,
    float hidden,
    float* log_coeff_out,
    float* log_value_out
) {
    // Save 1 softplus call: softplus(x) = x + softplus(-x) for any x
    float abs_gate = fabsf(gate);
    float sp_neg = log1pf(expf(-abs_gate));  // softplus(-|gate|) - single transcendental!
    float softplus_gate, softplus_neg_gate;
    if (gate >= 0.0f) {
        softplus_gate = gate + sp_neg;
        softplus_neg_gate = sp_neg;
    } else {
        softplus_gate = sp_neg;
        softplus_neg_gate = -gate + sp_neg;
    }

    *log_coeff_out = -softplus_gate;
    float log_z = -softplus_neg_gate;

    float log_tilde_h;
    if (hidden >= 0.0f) {
        log_tilde_h = logf(hidden + 0.5f);
    } else {
        log_tilde_h = -softplus_fwd(-hidden);
    }
    *log_value_out = log_z + log_tilde_h;
}

// Backward: computes gradients for gate and hidden
// given gradients for log_coeffs and log_values
__device__ __forceinline__ void log_coeffs_and_values_bwd(
    float grad_log_coeffs,
    float grad_log_values,
    float gate,
    float hidden,
    float* grad_gate_out,
    float* grad_hidden_out
) {
    // Optimization: sigmoid(-x) = 1 - sigmoid(x), so compute sigmoid(gate) once
    // softplus'(x) = sigmoid(x), so:
    //   d(-softplus(g))/dg = -sigmoid(g)
    //   d(-softplus(-g))/dg = sigmoid(-g) = 1 - sigmoid(g)
    float sig_gate = sigmoid(gate);
    *grad_gate_out = -grad_log_coeffs*sig_gate + grad_log_values*(1.0f - sig_gate);

    // grad_hidden from log_tilde_h
    if (hidden >= 0.0f) {
        // log_tilde_h = log(hidden + 0.5)
        // d(log_tilde_h)/d(hidden) = 1/(hidden + 0.5)
        *grad_hidden_out = grad_log_values / (hidden + 0.5f);
    } else {
        // log_tilde_h = -softplus(-hidden)
        // d(-softplus(-h))/dh = sigmoid(-h)
        *grad_hidden_out = grad_log_values * sigmoid(-hidden);
    }
}
