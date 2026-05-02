"""Probe whether the CUDA gsplat renderer actually propagates gradients.

The Sprint 4 smoke run on 3080 Ti showed model_psnr flat at ~12 dB across
4500 steps. CHANGELOG flags two failing CUDA backward tests in gsplat 1.4.0.
This script runs a single train step on CUDA and prints the gradient L2 norm
of every leaf parameter — if the renderer's backward is silent, every grad
norm will be zero (or near-zero from indirect paths only).

Run on the 3080 Ti machine inside the image-gs env:
    python scripts/probe_cuda_grad_flow.py
"""
from __future__ import annotations
import sys
import torch
import torch.nn.functional as F

from oss.gaussian.network import (
    CovariancePriorBank,
    GaussianParamNetwork,
    OutputHead,
)
from oss.gaussian.renderer import Rasterizer


def _grad_summary(name: str, param: torch.Tensor) -> str:
    if param.grad is None:
        return f"  {name:30s} grad=None"
    g = param.grad.detach()
    n = float(g.norm().item())
    nz = float((g != 0).float().mean().item())
    return f"  {name:30s} grad_norm={n:.6e}  nonzero_frac={nz:.4f}  shape={tuple(g.shape)}"


def main() -> int:
    device = "cuda"
    if not torch.cuda.is_available():
        print("FAIL: CUDA not available")
        return 1

    torch.manual_seed(0)
    bank = CovariancePriorBank(learnable=True)
    net = GaussianParamNetwork(bank_size=bank.bank_size, k_per_tile=3,
                               channels=(8, 16, 24, 32))
    head = OutputHead(bank=bank, tile_size=net.tile_size, k_per_tile=net.k_per_tile,
                      enable_gbuffer_bias=True)

    net.to(device)
    bank.to(device)
    head.to(device)

    # Break head zero-init so any backward signal can move weights.
    with torch.no_grad():
        net.head.weight.normal_(0, 0.01)
        net.head.bias.normal_(0, 0.01)

    H_lr, W_lr = 64, 64
    H_hr, W_hr = 128, 128
    x = torch.randn(2, 12, H_lr, W_lr, device=device)
    target = torch.rand(2, 3, H_hr, W_hr, device=device)
    depth = torch.rand(2, 1, H_lr, W_lr, device=device)
    normals = torch.randn(2, 3, H_lr, W_lr, device=device)
    normals = normals / normals.norm(dim=1, keepdim=True).clamp(min=1e-6)

    print(f"Device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    for backend_label in ("reference", "cuda"):
        print()
        print(f"=== Backend: {backend_label} ===")
        # Reset gradients on every parameter.
        for p in list(net.parameters()) + list(bank.parameters()) + list(head.parameters()):
            p.grad = None

        # For reference backend, drop to small resolution that fits VRAM.
        if backend_label == "reference":
            H_lr_b, W_lr_b = 32, 32
            H_hr_b, W_hr_b = 64, 64
            x_b = x[:, :, :H_lr_b, :W_lr_b].contiguous()
            target_b = target[:, :, :H_hr_b, :W_hr_b].contiguous()
            depth_b = depth[:, :, :H_lr_b, :W_lr_b].contiguous()
            normals_b = normals[:, :, :H_lr_b, :W_lr_b].contiguous()
        else:
            H_lr_b, W_lr_b, H_hr_b, W_hr_b = H_lr, W_lr, H_hr, W_hr
            x_b, target_b, depth_b, normals_b = x, target, depth, normals

        renderer = Rasterizer(force_backend=backend_label)
        raw = net(x_b)

        rendered_batch = []
        for b in range(x_b.shape[0]):
            gaussians = head.to_gaussian_batch(
                raw, batch_index=b,
                depth=depth_b[b:b+1], normals=normals_b[b:b+1],
            )
            rendered_batch.append(renderer(gaussians, output_hw=(H_hr_b, W_hr_b)))
        rendered = torch.stack(rendered_batch, dim=0)

        loss = F.mse_loss(rendered, target_b)
        print(f"  loss={float(loss.item()):.6f}")
        loss.backward()

        # Inspect gradient norms on representative leaves.
        report = [
            _grad_summary("net.stem.conv.weight", net.stem.conv.weight),
            _grad_summary("net.head.weight", net.head.weight),
            _grad_summary("head.gbuffer_bias.proj.weight",
                          head.gbuffer_bias.proj.weight),
            _grad_summary("bank.log_sx", bank.log_sx),
        ]
        for line in report:
            print(line)

        zero_grads = sum(
            1 for p in (
                net.stem.conv.weight,
                net.head.weight,
                head.gbuffer_bias.proj.weight,
                bank.log_sx,
            )
            if p.grad is None or float(p.grad.norm().item()) < 1e-12
        )
        if zero_grads > 0:
            print(f"  WARN: {zero_grads}/4 representative leaves had ~zero gradient")
        else:
            print(f"  OK: all 4 representative leaves received non-zero gradient")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
