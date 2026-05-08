"""Phase-1 smoke equivalence test: stub kernel == reference."""

import pytest
import torch

pytestmark = pytest.mark.cuda


def test_phase1_smoke_equivalence(cuda_device, kernels_built):
    from oss.cuda.oss_cuda import rasterize_gaussians
    from oss.gaussian.renderer.rasterizer import GaussianBatch, Rasterizer

    H, W, F = 32, 32, 3
    xy = torch.tensor([[16.0, 16.0]], device=cuda_device, dtype=torch.float32)
    scale = torch.tensor([[3.0, 3.0]], device=cuda_device, dtype=torch.float32)
    rot = torch.tensor([0.0], device=cuda_device, dtype=torch.float32)
    feat = torch.tensor([[1.0, 0.5, 0.25]], device=cuda_device, dtype=torch.float32)

    out_kernel = rasterize_gaussians(xy, scale, rot, feat, H, W, tile_size=16, topk_norm=True)

    rast = Rasterizer(tile_size=16, topk_norm=True)
    out_ref = rast._render_reference(GaussianBatch(xy=xy, scale=scale, rot=rot, feat=feat), H, W)

    assert out_kernel.shape == out_ref.shape, (
        f"shape mismatch: {out_kernel.shape} vs {out_ref.shape}"
    )
    assert out_kernel.shape == (F, H, W)
    torch.testing.assert_close(out_kernel, out_ref, atol=1e-5, rtol=1e-5)
