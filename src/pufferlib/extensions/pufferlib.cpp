#include <ATen/Operators.h>
#include <Python.h>
#include <torch/all.h>
#include <torch/library.h>
#include <vector>

extern "C" {
/* Creates a dummy empty _C module that can be imported from Python.
   The import from Python will load the .so consisting of this file
   in this extension, so that the TORCH_LIBRARY static initializers
   below are run. */
PyObject *PyInit__C(void) {
  static struct PyModuleDef module_def = {
      PyModuleDef_HEAD_INIT,
      "_C", /* name of module */
      NULL, /* module documentation, may be NULL */
      -1,   /* size of per-interpreter state of the module,
               or -1 if the module keeps state in global variables. */
      NULL, /* methods */
  };
  return PyModule_Create(&module_def);
}
}

namespace pufferlib {

void puff_advantage_row(float *values, float *rewards, float *dones,
                        float *importance, float *advantages, float gamma,
                        float lambda, float rho_clip, float c_clip,
                        int horizon) {
  float lastpufferlam = 0;
  for (int t = horizon - 2; t >= 0; t--) {
    int t_next = t + 1;
    float nextnonterminal = 1.0 - dones[t_next];
    float rho_t = fminf(importance[t], rho_clip);
    float c_t = fminf(importance[t], c_clip);
    float delta =
        rho_t * (rewards[t_next] + gamma * values[t_next] * nextnonterminal -
                 values[t]);
    lastpufferlam =
        delta + gamma * lambda * c_t * lastpufferlam * nextnonterminal;
    advantages[t] = lastpufferlam;
  }
}

void vtrace_check(torch::Tensor values, torch::Tensor rewards,
                  torch::Tensor dones, torch::Tensor importance,
                  torch::Tensor advantages, int num_steps, int horizon) {

  auto check = [&](const torch::Tensor &t, const char *name) {
    TORCH_CHECK(t.dim() == 2, name, " must be 2D");
    TORCH_CHECK(t.device() == values.device(),
                "All tensors must be on same device");
    TORCH_CHECK(t.size(0) == num_steps, name,
                ": first dimension must match num_steps");
    TORCH_CHECK(t.size(1) == horizon, name,
                ": second dimension must match horizon");
    TORCH_CHECK(t.dtype() == torch::kFloat32, "All tensors must be float32");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
  };

  // Validate input tensors
  check(values, "values");
  check(rewards, "rewards");
  check(dones, "dones");
  check(importance, "importance");
  check(advantages, "advantages");
}

// [num_steps, horizon]
void puff_advantage(float *values, float *rewards, float *dones,
                    float *importance, float *advantages, float gamma,
                    float lambda, float rho_clip, float c_clip, int num_steps,
                    const int horizon) {
  for (int offset = 0; offset < num_steps * horizon; offset += horizon) {
    puff_advantage_row(values + offset, rewards + offset, dones + offset,
                       importance + offset, advantages + offset, gamma, lambda,
                       rho_clip, c_clip, horizon);
  }
}

void compute_puff_advantage_cpu(torch::Tensor values, torch::Tensor rewards,
                                torch::Tensor dones, torch::Tensor importance,
                                torch::Tensor advantages, double gamma,
                                double lambda, double rho_clip, double c_clip) {
  int num_steps = values.size(0);
  int horizon = values.size(1);
  vtrace_check(values, rewards, dones, importance, advantages, num_steps,
               horizon);
  puff_advantage(values.data_ptr<float>(), rewards.data_ptr<float>(),
                 dones.data_ptr<float>(), importance.data_ptr<float>(),
                 advantages.data_ptr<float>(), gamma, lambda, rho_clip, c_clip,
                 num_steps, horizon);
}

TORCH_LIBRARY(pufferlib, m) {
  m.def("compute_puff_advantage(Tensor(a!) values, Tensor(b!) rewards, "
        "Tensor(c!) dones, Tensor(d!) importance, Tensor(e!) advantages, float "
        "gamma, float lambda, float rho_clip, float c_clip) -> ()");
  m.def("mingru_gate(Tensor state, Tensor combined) -> (Tensor, Tensor)");
  m.def("fc_max(Tensor x, Tensor W, Tensor b) -> Tensor");
  m.def("logcumsumexp_cuda(Tensor x) -> Tensor");
  // PufferLib 4.0 fused CUDA ops (CUDA-only impls; schema is always
  // registered).
  m.def("sample_logits(Tensor logits, Tensor logstd, Tensor value, Tensor "
        "actions_out, Tensor logprobs_out, Tensor value_out, Tensor act_sizes, "
        "int seed, Tensor offset) -> ()");
  m.def("fused_scan_checkpointed(Tensor combined, Tensor state) -> (Tensor, "
        "Tensor)");
  m.def("fused_ppo_loss_optimized(Tensor logits, Tensor logstd, Tensor "
        "values_pred, Tensor actions, Tensor old_logprobs, Tensor advantages, "
        "Tensor prio, Tensor values, Tensor returns, Tensor adv_mean, Tensor "
        "adv_var, Tensor ratio_out, Tensor newvalue_out, Tensor act_sizes, "
        "float clip_coef, float vf_clip_coef, float vf_coef, float ent_coef) "
        "-> Tensor");
}

TORCH_LIBRARY_IMPL(pufferlib, CPU, m) {
  m.impl("compute_puff_advantage", &compute_puff_advantage_cpu);
}

} // namespace pufferlib
