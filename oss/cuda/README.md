# OSS CUDA Extension

Phase 1 validates the custom-extension build, install, import, and equivalence
test chain before native CUDA kernels are added.

This package builds `oss_cuda._C` with a C++/pybind11 binding only. There are no
`.cu` sources in Phase 1. The rasterizer forward entry point calls through to
the Python reference wrapper, which renders via
`oss.gaussian.renderer.rasterizer.Rasterizer._render_reference`.

## Build

```bash
pip install -e ./oss/cuda
```

The build uses `torch.utils.cpp_extension.CppExtension`; `CUDAExtension` and
nvcc flags are intentionally deferred to Phase 2.

## Test

```bash
pytest tests/cuda/ -m cuda -v
```

Expected Phase 1 result: one CUDA smoke equivalence test passes with fp32 output
shape `(3, 32, 32)` and `torch.testing.assert_close` at `1e-5` tolerances.

## Scope

Implemented now:

- C++ extension import and pybind11 linking.
- Forward-only rasterizer autograd wrapper.
- CUDA-host smoke equivalence test against the PyTorch reference.

Not implemented until later phases:

- Native rasterizer CUDA forward or backward kernels.
- Fused cross-attention kernels.
- Training enablement via `OSS_USE_CUDA_KERNELS`.
