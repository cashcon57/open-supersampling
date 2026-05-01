# OSS-Gaussian Metal Port (M3 Max)

Sprint 7 / Track M scaffold. Ports the Sprint 1 CUDA tile rasterizer to MSL
and exports the Sprint 4 `GaussianParamNetwork` to CoreML.

## Build prerequisites (macOS arm64)

1. **Xcode Command Line Tools:** `xcode-select --install`
2. **Verify Metal compiler:** `xcrun -sdk macosx metal --version` should report Metal ≥ 32.x.
3. **Python extras (for CoreML export only):** `pip install -e '.[coreml]'` from repo root.

We deliberately do **not** depend on `metal-cpp`. The Swift host harness uses
the system Metal framework directly via the macOS SDK shipped with Xcode.

## Build

```sh
cd oss/gaussian/ports/metal
make            # builds rasterizer.metallib
make harness    # builds the Swift host harness (optional, for kernel parity tests)
```

The `metallib` is the artifact consumed by the Python driver (via PyObjC) at
runtime.

## Files

| File | Purpose |
| --- | --- |
| `rasterizer.metal` | MSL compute kernel skeleton. Body is a TODO; ported in T7.M.2. |
| `rasterizer.swift` | Swift host harness — loads the metallib, dispatches the kernel. |
| `export_coreml.py` | Converts a Sprint 4 checkpoint to `.mlpackage` via `coremltools`. |
| `Makefile` | Builds `rasterizer.metallib` and the Swift harness. |
| `__init__.py` | Python package init + `is_supported()` host check. |

## Run a CoreML export (dry-run, no `coremltools` required)

```sh
python -m oss.gaussian.ports.metal.export_coreml --check
```

Produces only a parity smoke-test number. Real export (writes
`checkpoints/param_net_lite.mlpackage`) requires `coremltools` ≥ 8.0:

```sh
python -m oss.gaussian.ports.metal.export_coreml --tier lite --ckpt path/to/sprint4.ckpt
```

## Sprint 7 status

Scaffold only. Empty kernel; `make` is expected to compile it cleanly. Real
kernel port + integration land in the Sprint 7 implementation phase per
`docs/superpowers/plans/2026-05-01-gaussian-sprint-7-plan.md`.
