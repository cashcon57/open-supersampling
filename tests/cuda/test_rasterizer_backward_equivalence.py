"""CUDA rasterizer backward equivalence tests."""

import pytest
import torch

pytestmark = pytest.mark.cuda


@pytest.mark.parametrize("N", [1, 16, 256, 4096])
@pytest.mark.parametrize(
    "H,W", [(32, 32), (64, 128), (256, 256), (270, 480), (540, 960)]
)
@pytest.mark.parametrize("F", [1, 3, 12, 64])
def test_rasterizer_backward_equivalence(cuda_device, kernels_built, N, H, W, F):
    from oss.cuda.oss_cuda import rasterize_gaussians
    from oss.gaussian.renderer.rasterizer import GaussianBatch, Rasterizer

    if N * max(H, 1) * max(W, 1) * max(F, 1) > 270_000_000:
        pytest.skip("too large for fast suite")

    torch.manual_seed(0xC0DA)
    xy = torch.rand(N, 2, device=cuda_device) * torch.tensor(
        [float(W), float(H)], device=cuda_device
    )
    scale = torch.rand(N, 2, device=cuda_device) * 5.0 + 0.5
    rot = torch.rand(N, device=cuda_device) * 6.28
    feat_kernel = torch.randn(N, F, device=cuda_device, requires_grad=True)
    feat_ref = feat_kernel.detach().clone().requires_grad_(True)

    out_kernel = rasterize_gaussians(
        xy, scale, rot, feat_kernel, H, W, tile_size=16, topk_norm=True
    )

    rast = Rasterizer(tile_size=16, topk_norm=True)
    out_ref = rast._render_reference(
        GaussianBatch(xy=xy, scale=scale, rot=rot, feat=feat_ref), H, W
    )
    assert out_kernel.shape == out_ref.shape == (F, H, W)

    grad_out = torch.randn_like(out_kernel)
    out_kernel.backward(grad_out)
    out_ref.backward(grad_out)

    assert feat_kernel.grad is not None
    assert feat_ref.grad is not None
    torch.testing.assert_close(
        feat_kernel.grad,
        feat_ref.grad,
        atol=1e-4,
        rtol=1e-4,
    )
