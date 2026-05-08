# CUDA Phase 3d Debug Memo - 2026-05-08

## Verdict

Phase 3d is blocked before the 100-step parity gate. I did not launch
`srcnn-v6.1-cuda-001`, because the 3080 Ti host cannot currently run the
Phase 3c native rasterizer backward.

## Findings

- Live PyTorch baseline trainer remains untouched: PID `22712`, run
  `E:\checkpoints\srcnn-v6.1-pico-001`.
- Active baseline launch command:
  `C:\Users\cashc\Miniconda3\envs\image-gs\python.exe scripts\sr_train_v6.py --output-dir E:\checkpoints\srcnn-v6.1-pico-001 --tartanair-root E:\datasets\tartanair_extracted --backbone hat-tiny --max-steps 300000 --warmup-steps 20000 --T0 50000 --num-restarts 5 --first-ckpt-step 100 --ckpt-every 500 --batch-size 4 --grad-accum 4 --patch-size 128 --trajectory-length 4 --num-workers 8 --spawn-offset-random --rasterizer-overlap 8`
- `E:\oss-gaussian` is the live training checkout, but it is at `6b3d94d` and
  has no `oss/cuda` package. I did not modify it.
- Created separate source checkout `E:\oss-gaussian-cuda-phase3d` at
  `6d7d354` and applied the missing `OSS_USE_CUDA_KERNELS=rasterizer` renderer
  gate there only.
- The installed extension in the `image-gs` environment loads from
  `E:\oss-gaussian-server\oss\cuda\oss_cuda\_C.cp311-win_amd64.pyd`, but it has
  only `rasterize_forward`; it lacks `rasterize_backward` and
  `conic_to_scale_rot_grad`.
- Rebuilding `oss/cuda` failed because `CUDA_HOME` is unset and no CUDA Toolkit
  / `nvcc` is installed in the standard location. `cl.exe` was also not found.

## Divergence Pattern

No loss divergence pattern exists yet. The kernel-backed trainer was not
started because the host would fail on the first backward pass with the
currently installed Phase 2-era extension.

## Baseline CSV

Baseline first-100 loss values are recorded in
`docs/coordination/cuda-phase3d-pico-first100-loss.csv`.

## Required Fix

Install the Windows CUDA build toolchain on the 3080 Ti host, including:

- CUDA Toolkit matching the PyTorch CUDA ABI closely enough for extension build
  (`torch 2.4.1`, CUDA runtime `12.4` on this host).
- MSVC Build Tools with `cl.exe` available to the build environment.

Then rebuild from `E:\oss-gaussian-cuda-phase3d`:

```powershell
$env:CUDA_HOME = '<CUDA toolkit root>'
$env:CUDA_PATH = $env:CUDA_HOME
$env:PATH = "$env:CUDA_HOME\bin;$env:PATH"
C:\Users\cashc\Miniconda3\envs\image-gs\python.exe -m pip install --no-build-isolation -e .\oss\cuda --force-reinstall
```

After import confirms all three symbols:

- `rasterize_forward`
- `rasterize_backward`
- `conic_to_scale_rot_grad`

rerun the 100-step gate before launching any 1k parity run.
