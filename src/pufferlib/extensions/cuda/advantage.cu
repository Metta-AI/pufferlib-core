#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/util/BFloat16.h>

namespace pufferlib {

// TIn = input type (bf16 or float), TOut = output type (always float for precision)
template<typename TIn, typename TOut>
__host__ __device__ void puff_advantage_row_cuda(const TIn* values, const TIn* rewards, const TIn* dones,
        const TIn* importance, TOut* advantages, float gamma, float lambda,
        float rho_clip, float c_clip, int horizon) {
    float lastpufferlam = 0;
    for (int t = horizon-2; t >= 0; t--) {
        int t_next = t + 1;
        float nextnonterminal = 1.0f - float(dones[t_next]);
        float imp = float(importance[t]);
        float rho_t = fminf(imp, rho_clip);
        float c_t = fminf(imp, c_clip);
        float delta = rho_t*(float(rewards[t_next]) + gamma*float(values[t_next])*nextnonterminal - float(values[t]));
        lastpufferlam = delta + gamma*lambda*c_t*lastpufferlam*nextnonterminal;
        advantages[t] = TOut(lastpufferlam);
    }
}

void vtrace_check_cuda(torch::Tensor values, torch::Tensor rewards,
        torch::Tensor dones, torch::Tensor importance, torch::Tensor advantages,
        int num_steps, int horizon) {

    auto check = [&](const torch::Tensor& t, const char* name) {
        TORCH_CHECK(t.dim() == 2, name, " must be 2D");
        TORCH_CHECK(t.device() == values.device(), "All tensors must be on same device");
        TORCH_CHECK(t.size(0) == num_steps, name, ": first dimension must match num_steps");
        TORCH_CHECK(t.size(1) == horizon, name, ": second dimension must match horizon");
        TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
    };

    // Validate input tensors
    auto input_dtype = values.dtype();
    check(values, "values");
    TORCH_CHECK(values.dtype() == input_dtype, "Input tensors must have matching dtype");
    check(rewards, "rewards");
    TORCH_CHECK(rewards.dtype() == input_dtype, "Input tensors must have matching dtype");
    check(dones, "dones");
    TORCH_CHECK(dones.dtype() == input_dtype, "Input tensors must have matching dtype");
    check(importance, "importance");
    TORCH_CHECK(importance.dtype() == input_dtype, "Input tensors must have matching dtype");

    // advantages can be different dtype (fp32 for precision)
    check(advantages, "advantages");
}

template<typename TIn, typename TOut>
__global__ void puff_advantage_kernel(const TIn* values, const TIn* rewards,
        const TIn* dones, const TIn* importance, TOut* advantages, float gamma,
        float lambda, float rho_clip, float c_clip, int num_steps, int horizon) {
    int row = blockIdx.x*blockDim.x + threadIdx.x;
    if (row >= num_steps) {
        return;
    }
    int offset = row*horizon;
    puff_advantage_row_cuda<TIn, TOut>(values + offset, rewards + offset, dones + offset,
        importance + offset, advantages + offset, gamma, lambda, rho_clip, c_clip, horizon);
}

template<typename TIn, typename TOut>
void compute_puff_advantage_cuda_impl(torch::Tensor values, torch::Tensor rewards,
        torch::Tensor dones, torch::Tensor importance, torch::Tensor advantages,
        double gamma, double lambda, double rho_clip, double c_clip) {
    int num_steps = values.size(0);
    int horizon = values.size(1);
    vtrace_check_cuda(values, rewards, dones, importance, advantages, num_steps, horizon);
    TORCH_CHECK(values.is_cuda(), "All tensors must be on GPU");

    int threads_per_block = 256;
    int blocks = (num_steps + threads_per_block - 1) / threads_per_block;

    puff_advantage_kernel<TIn, TOut><<<blocks, threads_per_block>>>(
        values.data_ptr<TIn>(),
        rewards.data_ptr<TIn>(),
        dones.data_ptr<TIn>(),
        importance.data_ptr<TIn>(),
        advantages.data_ptr<TOut>(),
        static_cast<float>(gamma),
        static_cast<float>(lambda),
        static_cast<float>(rho_clip),
        static_cast<float>(c_clip),
        num_steps,
        horizon
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
}

void compute_puff_advantage_cuda(torch::Tensor values, torch::Tensor rewards,
        torch::Tensor dones, torch::Tensor importance, torch::Tensor advantages,
        double gamma, double lambda, double rho_clip, double c_clip) {
    auto input_dtype = values.dtype();
    auto output_dtype = advantages.dtype();
    
    // Support bf16 inputs with fp32 output for precision
    if (input_dtype == torch::kFloat32 && output_dtype == torch::kFloat32) {
        compute_puff_advantage_cuda_impl<float, float>(values, rewards, dones, importance, advantages,
            gamma, lambda, rho_clip, c_clip);
    } else if (input_dtype == torch::kBFloat16 && output_dtype == torch::kFloat32) {
        compute_puff_advantage_cuda_impl<at::BFloat16, float>(values, rewards, dones, importance, advantages,
            gamma, lambda, rho_clip, c_clip);
    } else if (input_dtype == torch::kBFloat16 && output_dtype == torch::kBFloat16) {
        compute_puff_advantage_cuda_impl<at::BFloat16, at::BFloat16>(values, rewards, dones, importance, advantages,
            gamma, lambda, rho_clip, c_clip);
    } else {
        TORCH_CHECK(false, "Unsupported dtype combination: inputs must be float32 or bfloat16, advantages must be float32 or bfloat16");
    }
}

TORCH_LIBRARY_IMPL(pufferlib, CUDA, m) {
  m.impl("compute_puff_advantage", &compute_puff_advantage_cuda);
}

}
