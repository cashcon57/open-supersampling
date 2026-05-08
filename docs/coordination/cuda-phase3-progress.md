# CUDA Phase 3 Progress

## 2026-05-08 - Phase 3d parity acceptance

**Status:** blocked before 100-step gate.

Phase 3d was prepared as a separate run named `srcnn-v6.1-cuda-001` with output
directory `E:\checkpoints\srcnn-v6.1-cuda-001`. The current
`srcnn-v6.1-pico-001` trainer, PID `22712`, was not touched.

The active baseline trainer was launched from `E:\oss-gaussian` with:

```powershell
C:\Users\cashc\Miniconda3\envs\image-gs\python.exe scripts\sr_train_v6.py --output-dir E:\checkpoints\srcnn-v6.1-pico-001 --tartanair-root E:\datasets\tartanair_extracted --backbone hat-tiny --max-steps 300000 --warmup-steps 20000 --T0 50000 --num-restarts 5 --first-ckpt-step 100 --ckpt-every 500 --batch-size 4 --grad-accum 4 --patch-size 128 --trajectory-length 4 --num-workers 8 --spawn-offset-random --rasterizer-overlap 8
```

For the CUDA run, I created a separate source checkout at
`E:\oss-gaussian-cuda-phase3d` from `6d7d354` and applied the missing renderer
feature gate so `OSS_USE_CUDA_KERNELS=rasterizer` actually selects
`oss.cuda.oss_cuda.rasterize_gaussians`.

The 100-step WMI orphan-spawn was not launched. The 3080 Ti host currently has
CUDA-enabled PyTorch (`torch 2.4.1`, runtime `12.4`, RTX 3080 Ti), but not the
CUDA/MSVC build toolchain required to build the Phase 3c extension:

- no CUDA Toolkit found at `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA`
- `where nvcc` found no compiler
- `where cl` found no MSVC compiler
- installed `oss_cuda._C` loads from `E:\oss-gaussian-server`, but lacks
  `rasterize_backward` and `conic_to_scale_rot_grad`

**Loss CSV:** `docs/coordination/cuda-phase3d-pico-first100-loss.csv`

**Parity verdict:** not run / blocked. There are no CUDA losses to compare.

**Default-on recommendation:** do not flip default-on. Install the CUDA/MSVC
build toolchain or provide a Phase 3c-compatible prebuilt `oss_cuda._C` artifact,
then rerun the 100-step ±0.5% gate before launching the 1k ±1% acceptance run.

Debug details are in `docs/coordination/cuda-phase3-debug-2026-05-08.md`.
