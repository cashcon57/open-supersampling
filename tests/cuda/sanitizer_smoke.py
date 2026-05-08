"""Compute Sanitizer smoke for the Phase 3c rasterizer."""

from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _run_shape(n: int, h: int, w: int, f: int) -> None:
    torch.manual_seed(0xC0DA)
    if n == 0:
        xy_cpu = torch.empty((0, 2), dtype=torch.float32)
        scale_cpu = torch.empty((0, 2), dtype=torch.float32)
        rot_cpu = torch.empty((0,), dtype=torch.float32)
        feat_cpu = torch.empty((0, f), dtype=torch.float32)
    else:
        idx = torch.arange(n, dtype=torch.float32)
        xy_cpu = torch.stack(
            [
                torch.remainder(idx * 13.0 + 7.0, float(w)),
                torch.remainder(idx * 17.0 + 11.0, float(h)),
            ],
            dim=1,
        )
        scale_cpu = torch.stack(
            [
                1.0 + torch.remainder(idx, 7.0),
                1.5 + torch.remainder(idx * 3.0, 11.0),
            ],
            dim=1,
        )
        rot_cpu = torch.remainder(idx, 23.0) * 0.071
        feat_idx = torch.arange(f, dtype=torch.float32)
        feat_cpu = torch.sin(idx[:, None] * 0.013 + feat_idx[None, :] * 0.17)

    device = torch.device("cuda:0")
    xy = xy_cpu.to(device).requires_grad_(True)
    scale = scale_cpu.to(device).requires_grad_(True)
    rot = rot_cpu.to(device).requires_grad_(True)
    feat = feat_cpu.to(device).requires_grad_(True)

    from oss.cuda.oss_cuda import rasterize_gaussians

    out = rasterize_gaussians(xy, scale, rot, feat, h, w, 16, True)
    grad_out = torch.ones(out.shape, dtype=out.dtype).to(device)
    out.backward(grad_out)
    torch.cuda.synchronize()
    assert out.shape == (f, h, w)
    assert torch.isfinite(out.cpu()).all()
    assert xy.grad is not None
    assert torch.isfinite(xy.grad.cpu()).all()
    assert feat.grad is not None
    assert torch.isfinite(feat.grad.cpu()).all()
    assert scale.grad is not None
    assert torch.isfinite(scale.grad.cpu()).all()
    assert rot.grad is not None
    assert torch.isfinite(rot.grad.cpu()).all()
    d_xy_sum = xy.grad.detach().cpu().sum().item()
    d_scale_sum = scale.grad.detach().cpu().sum().item()
    d_rot_sum = rot.grad.detach().cpu().sum().item()
    d_feat_sum = feat.grad.detach().cpu().sum().item()
    print(f"smoke backward d_xy sum: {d_xy_sum:.6f}")
    print(f"smoke backward d_scale sum: {d_scale_sum:.6f}")
    print(f"smoke backward d_rot sum: {d_rot_sum:.6f}")
    print(f"smoke backward d_feat sum: {d_feat_sum:.6f}")


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device required")
    _run_shape(16, 64, 64, 3)
    _run_shape(512, 270, 480, 12)


if __name__ == "__main__":
    main()
