#include <torch/library.h>

#include "modules.h"

// Registers CUDA-only kernels under torch.ops.pufferlib.*.
//
// Note: We intentionally do not provide CPU implementations. If a caller hits
// these ops on CPU, it should fail loudly.

static std::tuple<torch::Tensor, torch::Tensor>
mingru_gate_tuple(torch::Tensor state, torch::Tensor combined) {
  auto out = mingru_gate(std::move(state), std::move(combined));
  TORCH_CHECK(out.size() == 2,
              "mingru_gate must return exactly 2 tensors, got ", out.size());
  return {std::move(out[0]), std::move(out[1])};
}

static std::tuple<torch::Tensor, torch::Tensor>
fused_scan_checkpointed_tuple(torch::Tensor combined, torch::Tensor state) {
  auto out = fused_scan_checkpointed(std::move(combined), std::move(state));
  TORCH_CHECK(out.size() == 2,
              "fused_scan_checkpointed must return exactly 2 tensors, got ",
              out.size());
  return {std::move(out[0]), std::move(out[1])};
}

static torch::Tensor fused_ppo_loss_optimized_tensor(
    torch::Tensor logits, torch::Tensor logstd, torch::Tensor values_pred,
    torch::Tensor actions, torch::Tensor old_logprobs, torch::Tensor advantages,
    torch::Tensor prio, torch::Tensor values, torch::Tensor returns,
    torch::Tensor adv_mean, torch::Tensor adv_var, torch::Tensor ratio_out,
    torch::Tensor newvalue_out, torch::Tensor act_sizes, double clip_coef,
    double vf_clip_coef, double vf_coef, double ent_coef) {
  auto out = fused_ppo_loss_optimized(
      std::move(logits), std::move(logstd), std::move(values_pred),
      std::move(actions), std::move(old_logprobs), std::move(advantages),
      std::move(prio), std::move(values), std::move(returns),
      std::move(adv_mean), std::move(adv_var), std::move(ratio_out),
      std::move(newvalue_out), std::move(act_sizes),
      static_cast<float>(clip_coef), static_cast<float>(vf_clip_coef),
      static_cast<float>(vf_coef), static_cast<float>(ent_coef));
  TORCH_CHECK(out.size() == 1,
              "fused_ppo_loss_optimized must return exactly 1 tensor, got ",
              out.size());
  return std::move(out[0]);
}

static void sample_logits_void(torch::Tensor logits, torch::Tensor logstd,
                               torch::Tensor value, torch::Tensor actions_out,
                               torch::Tensor logprobs_out,
                               torch::Tensor value_out, torch::Tensor act_sizes,
                               int64_t seed, torch::Tensor offset) {
  sample_logits(std::move(logits), std::move(logstd), std::move(value),
                std::move(actions_out), std::move(logprobs_out),
                std::move(value_out), std::move(act_sizes),
                static_cast<uint64_t>(seed), std::move(offset));
}

TORCH_LIBRARY_IMPL(pufferlib, CUDA, m) {
  m.impl("mingru_gate", &mingru_gate_tuple);
  m.impl("fc_max", &fc_max);
  m.impl("logcumsumexp_cuda", &logcumsumexp_cuda);
  m.impl("sample_logits", &sample_logits_void);
  m.impl("fused_scan_checkpointed", &fused_scan_checkpointed_tuple);
  m.impl("fused_ppo_loss_optimized", &fused_ppo_loss_optimized_tensor);
}
