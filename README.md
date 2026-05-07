# pufferlib-core

Minimal PufferLib core functionality vendored into the Metta monorepo.

This package builds an optional `pufferlib._C` C++ extension, including CUDA
`torch.ops.pufferlib.*` kernels when a CUDA toolchain is available. Set
`FORCE_CUDA=1` to require CUDA kernels, or `FORCE_CUDA=0` to force a CPU-only
build.
