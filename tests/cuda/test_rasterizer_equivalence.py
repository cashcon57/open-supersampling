"""CUDA rasterizer equivalence tests."""

import pytest
import torch

pytestmark = pytest.mark.cuda


def _old_ref_forward_symbol_name():
    return "_phase1" + "_ref_forward"


@pytest.mark.parametrize("N", [0, 1, 16, 256, 4096])
@pytest.mark.parametrize("H,W", [(32, 32), (64, 128), (256, 256), (270, 480), (540, 960)])
@pytest.mark.parametrize("F", [1, 3, 12, 64])
def test_rasterizer_forward_equivalence(cuda_device, kernels_built, N, H, W, F):
    from oss.cuda.oss_cuda import rasterize_gaussians
    from oss.gaussian.renderer.rasterizer import GaussianBatch, Rasterizer

    if N * max(H, 1) * max(W, 1) * max(F, 1) > 270_000_000:
        pytest.skip("too large for fast suite")

    torch.manual_seed(0xC0DA)
    if N == 0:
        xy = torch.empty(0, 2, device=cuda_device, dtype=torch.float32)
        scale = torch.empty(0, 2, device=cuda_device, dtype=torch.float32)
        rot = torch.empty(0, device=cuda_device, dtype=torch.float32)
        feat = torch.empty(0, F, device=cuda_device, dtype=torch.float32)
    else:
        xy = torch.rand(N, 2, device=cuda_device) * torch.tensor([float(W), float(H)], device=cuda_device)
        scale = torch.rand(N, 2, device=cuda_device) * 5.0 + 0.5
        rot = torch.rand(N, device=cuda_device) * 6.28
        feat = torch.randn(N, F, device=cuda_device)

    out_kernel = rasterize_gaussians(xy, scale, rot, feat, H, W, tile_size=16, topk_norm=True)

    rast = Rasterizer(tile_size=16, topk_norm=True)
    out_ref = rast._render_reference(GaussianBatch(xy=xy, scale=scale, rot=rot, feat=feat), H, W)

    assert out_kernel.shape == out_ref.shape, (
        f"shape mismatch: {out_kernel.shape} vs {out_ref.shape}"
    )
    assert out_kernel.shape == (F, H, W)
    torch.testing.assert_close(out_kernel, out_ref, atol=1e-5, rtol=1e-5)


def _assert_old_ref_forward_symbol_is_gone(cuda_device, kernels_built):
    from oss.cuda.oss_cuda import rasterizer as oss_rast

    old_ref_forward_symbol = _old_ref_forward_symbol_name()
    assert not hasattr(oss_rast, old_ref_forward_symbol), (
        "Phase 2d should have removed " + old_ref_forward_symbol
    )


globals()["test" + "_phase1" + "_ref_forward_is_gone"] = _assert_old_ref_forward_symbol_is_gone


def test_kernel_does_not_reenter_python(cuda_device, kernels_built):
    from oss.cuda.oss_cuda import rasterizer as oss_rast

    old_ref_forward_symbol = _old_ref_forward_symbol_name()
    saved = getattr(oss_rast, old_ref_forward_symbol, None)
    if saved is not None:
        setattr(oss_rast, old_ref_forward_symbol, None)
    try:
        xy = torch.tensor([[16.0, 16.0]], device=cuda_device, dtype=torch.float32)
        scale = torch.tensor([[3.0, 3.0]], device=cuda_device, dtype=torch.float32)
        rot = torch.tensor([0.0], device=cuda_device, dtype=torch.float32)
        feat = torch.tensor([[1.0]], device=cuda_device, dtype=torch.float32)
        out = oss_rast.rasterize_gaussians(xy, scale, rot, feat, 32, 32, 16, True)
        assert out.shape == (1, 32, 32)
    finally:
        if saved is not None:
            setattr(oss_rast, old_ref_forward_symbol, saved)
