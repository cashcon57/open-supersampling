# OSS CUDA Extension

Phase 2a adds the first native CUDA rasterizer preprocessing kernel while
keeping rasterization itself on the Python reference path.

This package builds `oss_cuda._C` as a prebuilt PyTorch extension. Phase 2a
exposes `_C.preprocess_only` for CUDA conic/AABB preprocessing; the
rasterizer forward entry point still calls through to the Python reference
wrapper for the final rasterized output.

## Build

```bash
pip install --no-build-isolation -e ./oss/cuda --force-reinstall
```

`--no-build-isolation` makes the extension use the torch/CUDA ABI from the
active environment, so install PyTorch before running this command.
`--force-reinstall` prevents stale build artifacts from hiding source changes.

## Test

```bash
pytest tests/cuda/ -m cuda -v
```

Expected Phase 2a result: the Phase 1 smoke equivalence test still passes, and
`tests/cuda/test_preprocess_kernel.py`
matches CUDA preprocess `conic`, AABB, and tile-count outputs against the
PyTorch oracle. The `conic` tolerance is `1e-6`; integer outputs are exact.

## C++ Kernel Tests

```bash
cd tests/cuda/cpp
cmake -B build -DCMAKE_PREFIX_PATH=$(python -c "import torch; print(torch.utils.cmake_prefix_path)")
cmake --build build
./build/test_oss_cuda
```

## Scope

Implemented now:

- C++ extension import and pybind11 linking.
- Native CUDA preprocess kernel for conic/AABB/tile-count scratch tensors.
- Forward-only rasterizer autograd wrapper.
- CUDA preprocess equivalence test against the PyTorch oracle.
- CUDA-host smoke equivalence test against the PyTorch reference.

Not implemented until later phases:

- Native rasterizer CUDA sum-composite or backward kernels.
- Fused cross-attention kernels.
- Training enablement via `OSS_USE_CUDA_KERNELS`.
