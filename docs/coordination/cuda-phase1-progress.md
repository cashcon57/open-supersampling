# CUDA Phase 1 Progress

Phase 1 built the OSS-native CUDA extension plumbing without native CUDA
kernels. The new `oss/cuda` package installs an `oss_cuda._C` C++ extension via
`CppExtension`; its rasterizer forward binding validates tensor shapes, then
delegates back to `Rasterizer._render_reference` through the Python
`_phase1_ref_forward` shim. The Python side exposes a forward-only autograd
Function for rasterization and a `NotImplementedError` cross-attention stub.

Verification ran on `3080ti-windows` (`CASH-PC`, RTX 3080 Ti, torch 2.4.1
cu124). `python -m pip install -e ./oss/cuda` completed, `_C` imported, and
`python -m pytest tests/cuda/ -m cuda -v` passed the smoke equivalence test:
`tests/cuda/test_rasterizer_equivalence.py::test_phase1_smoke_equivalence PASSED`.
The pass criteria were output shape `(3, 32, 32)` and `assert_close` at
`atol=1e-5, rtol=1e-5`.

No real rasterizer forward, rasterizer backward, or cross-attention CUDA kernel
was built; those remain Phase 2+. Build notes: setuptools rejected the original
`0.1.0-phase1` metadata version, so the package uses PEP 440 `0.1.0+phase1`.
Build isolation also needed torch pinned to the host ABI (`torch==2.4.1`) to
avoid a Windows `_C` import failure.
