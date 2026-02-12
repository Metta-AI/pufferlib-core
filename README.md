# pufferlib-core

Minimal PufferLib core functionality vendored into the Metta monorepo.

This package builds an optional `pufferlib._C` C++/CUDA extension that registers `torch.ops.pufferlib.*` kernels when
CUDA is available.
