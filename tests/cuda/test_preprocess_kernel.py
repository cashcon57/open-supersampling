import pytest
import torch

pytestmark = pytest.mark.cuda


def _ref_preprocess(xy, scale, rot, H, W, tile_size=16):
    cos_r = rot.cos()
    sin_r = rot.sin()
    sx = scale[:, 0].clamp_min(1e-6)
    sy = scale[:, 1].clamp_min(1e-6)
    inv_sx2 = 1.0 / (sx * sx)
    inv_sy2 = 1.0 / (sy * sy)
    a = cos_r * cos_r * inv_sx2 + sin_r * sin_r * inv_sy2
    b = cos_r * sin_r * (inv_sx2 - inv_sy2)
    d = sin_r * sin_r * inv_sx2 + cos_r * cos_r * inv_sy2
    conic = torch.stack([a, b, d], dim=-1)
    radius = 3.0 * torch.maximum(scale[:, 0], scale[:, 1])
    num_tx = (W + tile_size - 1) // tile_size
    num_ty = (H + tile_size - 1) // tile_size
    x_lo = ((xy[:, 0] - radius) / tile_size).floor().clamp_(0, num_tx).to(torch.int32)
    x_hi = ((xy[:, 0] + radius) / tile_size).ceil().clamp_(0, num_tx).to(torch.int32)
    y_lo = ((xy[:, 1] - radius) / tile_size).floor().clamp_(0, num_ty).to(torch.int32)
    y_hi = ((xy[:, 1] + radius) / tile_size).ceil().clamp_(0, num_ty).to(torch.int32)
    aabb = torch.stack([x_lo, y_lo, x_hi, y_hi], dim=-1)
    pair_count = ((x_hi - x_lo).clamp_min(0) * (y_hi - y_lo).clamp_min(0)).to(torch.int32)
    finite = a.isfinite() & b.isfinite() & d.isfinite()
    pair_count = torch.where(finite, pair_count, torch.zeros_like(pair_count))
    return conic, aabb, pair_count


@pytest.mark.parametrize("N", [1, 32, 256])
@pytest.mark.parametrize("H,W", [(128, 128), (270, 480), (540, 960)])
def test_preprocess_kernel_matches_pytorch_ref(cuda_device, kernels_built, N, H, W):
    torch.manual_seed(0xC0DA)
    xy = torch.rand(N, 2, device=cuda_device) * torch.tensor([float(W), float(H)], device=cuda_device)
    scale = torch.rand(N, 2, device=cuda_device) * 5.0 + 0.5
    rot = torch.rand(N, device=cuda_device) * 6.28
    from oss.cuda.oss_cuda._test_helpers import preprocess_only

    conic_k, aabb_k, pc_k = preprocess_only(xy, scale, rot, H, W, 16)
    conic_r, aabb_r, pc_r = _ref_preprocess(xy, scale, rot, H, W, 16)
    torch.testing.assert_close(conic_k, conic_r, atol=1e-6, rtol=1e-6)
    assert torch.equal(aabb_k, aabb_r), "aabb mismatch"
    assert torch.equal(pc_k, pc_r), "pair_count mismatch"
