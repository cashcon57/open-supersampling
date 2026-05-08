"""CUDA rasterizer backward equivalence tests."""

import pytest
import torch

pytestmark = pytest.mark.cuda


def _conic_from_scale_rot(scale, rot):
    cos_t = torch.cos(rot)
    sin_t = torch.sin(rot)
    sx = scale[:, 0].clamp(min=1e-6)
    sy = scale[:, 1].clamp(min=1e-6)
    inv_sx2 = 1.0 / (sx * sx)
    inv_sy2 = 1.0 / (sy * sy)
    a = cos_t * cos_t * inv_sx2 + sin_t * sin_t * inv_sy2
    b = cos_t * sin_t * (inv_sx2 - inv_sy2)
    d = sin_t * sin_t * inv_sx2 + cos_t * cos_t * inv_sy2
    return torch.stack((a, b, d), dim=-1).contiguous()


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
    xy_kernel = xy.detach().clone().requires_grad_(True)
    xy_ref = xy.detach().clone().requires_grad_(True)
    scale_kernel = scale.detach().clone().requires_grad_(True)
    rot_kernel = rot.detach().clone().requires_grad_(True)
    feat_kernel = torch.randn(N, F, device=cuda_device, requires_grad=True)
    feat_ref = feat_kernel.detach().clone().requires_grad_(True)

    out_kernel = rasterize_gaussians(
        xy_kernel, scale_kernel, rot_kernel, feat_kernel, H, W, tile_size=16, topk_norm=True
    )

    rast = Rasterizer(tile_size=16, topk_norm=True)
    out_ref = rast._render_reference(
        GaussianBatch(xy=xy_ref, scale=scale, rot=rot, feat=feat_ref), H, W
    )
    assert out_kernel.shape == out_ref.shape == (F, H, W)

    grad_out = torch.randn_like(out_kernel)
    out_kernel.backward(grad_out)
    out_ref.backward(grad_out)

    assert xy_kernel.grad is not None
    assert xy_ref.grad is not None
    assert feat_kernel.grad is not None
    assert feat_ref.grad is not None
    assert scale_kernel.grad is None
    assert rot_kernel.grad is None
    torch.testing.assert_close(
        xy_kernel.grad,
        xy_ref.grad,
        atol=1e-4,
        rtol=1e-4,
    )
    torch.testing.assert_close(
        feat_kernel.grad,
        feat_ref.grad,
        atol=1e-4,
        rtol=1e-4,
    )


def test_rasterizer_backward_dconic_matches_reference(cuda_device, kernels_built):
    from oss.cuda.oss_cuda.rasterizer import _C
    from oss.gaussian.renderer.rasterizer import GaussianBatch, Rasterizer

    torch.manual_seed(0xD0C0)
    N, H, W, F = 3, 11, 13, 17
    xy = torch.rand(N, 2, device=cuda_device) * torch.tensor(
        [float(W), float(H)], device=cuda_device
    )
    scale = torch.rand(N, 2, device=cuda_device) * 4.0 + 0.75
    rot = torch.rand(N, device=cuda_device) * 6.28
    feat = torch.randn(N, F, device=cuda_device)
    conic_kernel = _conic_from_scale_rot(scale, rot)
    conic_ref = conic_kernel.detach().clone().requires_grad_(True)

    _, gaussian_idx_sorted, tile_offsets, _ = _C.rasterize_forward(
        xy, scale, rot, feat, H, W, 16, True
    )
    grad_out = torch.randn(F, H, W, device=cuda_device)
    _, d_conic_kernel, _ = _C.rasterize_backward(
        xy,
        scale,
        rot,
        feat,
        conic_kernel,
        gaussian_idx_sorted,
        tile_offsets,
        grad_out,
        H,
        W,
        16,
    )

    rast = Rasterizer(tile_size=16, topk_norm=True)
    out_ref = rast._render_reference(
        GaussianBatch(xy=xy, scale=scale, rot=rot, feat=feat),
        H,
        W,
        conic=conic_ref,
    )
    out_ref.backward(grad_out)

    assert conic_ref.grad is not None
    torch.testing.assert_close(
        d_conic_kernel,
        conic_ref.grad,
        atol=1e-4,
        rtol=1e-4,
    )
