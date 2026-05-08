import pytest
import torch

from tests.cuda.test_preprocess_kernel import _ref_preprocess

pytestmark = pytest.mark.cuda


def _ref_pair_construction(xy, scale, rot, H, W, tile_size=16):
    conic, aabb, pair_count = _ref_preprocess(xy, scale, rot, H, W, tile_size)
    total_pairs = int(pair_count.sum().item())
    num_tx = (W + tile_size - 1) // tile_size
    num_ty = (H + tile_size - 1) // tile_size
    num_tiles = num_tx * num_ty

    if total_pairs == 0:
        gid_sorted = torch.empty((0,), device=xy.device, dtype=torch.int32)
        tile_offsets = torch.zeros((num_tiles + 1,), device=xy.device, dtype=torch.int32)
        return gid_sorted, tile_offsets, conic, total_pairs

    keys = []
    gids = []
    aabb_cpu = aabb.cpu().tolist()
    for gid, (x_lo, y_lo, x_hi, y_hi) in enumerate(aabb_cpu):
        for ty in range(y_lo, y_hi):
            for tx in range(x_lo, x_hi):
                tile_id = ty * num_tx + tx
                keys.append((tile_id << 32) | gid)
                gids.append(gid)

    keys_t = torch.tensor(keys, device=xy.device, dtype=torch.int64)
    gids_t = torch.tensor(gids, device=xy.device, dtype=torch.int32)
    keys_sorted, order = torch.sort(keys_t)
    gid_sorted = gids_t[order]

    boundaries = torch.arange(num_tiles + 1, device=xy.device, dtype=torch.int64) << 32
    tile_offsets = torch.searchsorted(keys_sorted, boundaries).to(torch.int32)
    return gid_sorted, tile_offsets, conic, total_pairs


def _total_pairs_to_int(total_pairs):
    if torch.is_tensor(total_pairs):
        return int(total_pairs.item())
    return int(total_pairs)


@pytest.mark.parametrize("N", [4, 16, 64])
@pytest.mark.parametrize("H,W", [(64, 64), (256, 256)])
def test_pair_construction_kernel_matches_pytorch_ref(cuda_device, kernels_built, N, H, W):
    torch.manual_seed(0xC0DA)
    xy = torch.rand(N, 2, device=cuda_device) * torch.tensor([float(W), float(H)], device=cuda_device)
    scale = torch.rand(N, 2, device=cuda_device) * 5.0 + 0.5
    rot = torch.rand(N, device=cuda_device) * 6.28
    from oss.cuda.oss_cuda._test_helpers import pair_construction_only

    gid_sorted_k, tile_offsets_k, conic_k, total_pairs_k = pair_construction_only(xy, scale, rot, H, W, 16)
    gid_sorted_r, tile_offsets_r, conic_r, total_pairs_r = _ref_pair_construction(xy, scale, rot, H, W, 16)

    assert _total_pairs_to_int(total_pairs_k) == total_pairs_r, "total_pairs mismatch"
    torch.testing.assert_close(conic_k, conic_r, atol=1e-6, rtol=1e-6)
    assert torch.equal(gid_sorted_k, gid_sorted_r), "gid_sorted mismatch"
    assert torch.equal(tile_offsets_k, tile_offsets_r), "tile_offsets mismatch"
