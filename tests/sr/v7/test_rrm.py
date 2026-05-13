"""Tests for Random Reshading Masking (RRM) — STSS-style disocclusion
augmentation.

RRM generates random LR patches, zeros motion vectors there, and applies
2x loss weight in the matching HR region. The model is forced to be
robust to "history-unavailable" regions, which is what disocclusion
looks like at inference time."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from sr_train_v7 import (
    random_reshading_mask, apply_rrm_to_sample, _LR_KEYS, _HR_KEYS,
)
from oss.sr.v7.losses import charbonnier, oss_fx_loss


def test_random_reshading_mask_shapes_correct():
    g = torch.Generator().manual_seed(0)
    lr_mask, hr_weight = random_reshading_mask(
        h_lr=32, w_lr=48, scale=2, n_patches=3, generator=g,
    )
    assert lr_mask.shape == (1, 1, 32, 48)
    assert lr_mask.dtype == torch.bool
    assert hr_weight.shape == (1, 1, 64, 96)
    assert hr_weight.dtype == torch.float32


def test_random_reshading_mask_default_loss_weight_2x_inside_patches():
    g = torch.Generator().manual_seed(1)
    lr_mask, hr_weight = random_reshading_mask(
        h_lr=32, w_lr=48, scale=2, n_patches=4,
        loss_weight=2.0, generator=g,
    )
    # Outside patches the weight is 1; inside it's the configured loss_weight.
    n_masked_lr = int(lr_mask.sum().item())
    assert n_masked_lr > 0
    weight_in_patch = float(hr_weight[hr_weight > 1.0].mean().item())
    weight_outside = float(hr_weight[hr_weight == 1.0].mean().item())
    assert weight_in_patch == 2.0
    assert weight_outside == 1.0


def test_random_reshading_mask_custom_loss_weight():
    g = torch.Generator().manual_seed(2)
    _, hr_weight = random_reshading_mask(
        h_lr=16, w_lr=16, scale=2, n_patches=2, loss_weight=5.0, generator=g,
    )
    max_w = float(hr_weight.max().item())
    assert max_w == 5.0


def test_random_reshading_mask_LR_HR_alignment():
    """A pixel that's masked at LR (top-left = (top_lr, left_lr)) should
    correspond to a weighted HR region from
    (top_lr*scale, left_lr*scale) to (... + patch_size*scale)."""
    g = torch.Generator().manual_seed(3)
    lr_mask, hr_weight = random_reshading_mask(
        h_lr=20, w_lr=30, scale=2, n_patches=1, patch_size_min=4, patch_size_max=4,
        loss_weight=2.0, generator=g,
    )
    # Find the patch in LR
    lr_idx = lr_mask[0, 0].nonzero()
    top_lr = int(lr_idx[:, 0].min().item())
    left_lr = int(lr_idx[:, 1].min().item())
    bottom_lr = int(lr_idx[:, 0].max().item()) + 1
    right_lr = int(lr_idx[:, 1].max().item()) + 1
    # Corresponding HR rect must have weight 2.0
    hr_box = hr_weight[0, 0, top_lr*2:bottom_lr*2, left_lr*2:right_lr*2]
    assert torch.all(hr_box == 2.0)


def test_random_reshading_mask_n_patches_is_respected():
    """n_patches=0 should produce an empty mask + all-1 weight."""
    g = torch.Generator().manual_seed(4)
    lr_mask, hr_weight = random_reshading_mask(
        h_lr=16, w_lr=16, scale=2, n_patches=0, generator=g,
    )
    assert int(lr_mask.sum().item()) == 0
    assert torch.all(hr_weight == 1.0)


def test_apply_rrm_zeros_motion_vectors_in_masked_lr_regions_only():
    """apply_rrm_to_sample should zero the LR motion buffer inside patches
    while leaving everything else (RGB, depth, normals, HR GT) untouched."""
    h_lr, w_lr = 16, 24
    sample = {
        "n_lr":           torch.rand((1, 3, h_lr, w_lr)),
        "n_depth":        torch.rand((1, 1, h_lr, w_lr)),
        "n_motion":       torch.rand((1, 2, h_lr, w_lr)) + 1.0,  # all > 1
        "n_normals":      torch.rand((1, 3, h_lr, w_lr)),
        "np1_lr":         torch.rand((1, 3, h_lr, w_lr)),
        "np1_depth":      torch.rand((1, 1, h_lr, w_lr)),
        "np1_motion":     torch.rand((1, 2, h_lr, w_lr)) + 1.0,
        "np1_normals":    torch.rand((1, 3, h_lr, w_lr)),
        "motion_n_to_np1": torch.rand((1, 2, h_lr, w_lr)) + 1.0,
        "n_gt":           torch.rand((1, 3, h_lr*2, w_lr*2)),
        "n_half_gt":      torch.rand((1, 3, h_lr*2, w_lr*2)),
        "np1_gt":         torch.rand((1, 3, h_lr*2, w_lr*2)),
    }
    g = torch.Generator().manual_seed(5)
    lr_mask, _ = random_reshading_mask(
        h_lr=h_lr, w_lr=w_lr, scale=2, n_patches=3, generator=g,
    )
    out = apply_rrm_to_sample(sample, lr_mask)

    # Motion buffers: zero where masked, unchanged where not
    mask_squashed = lr_mask[0, 0]
    assert torch.all(out["n_motion"][0, 0][mask_squashed] == 0.0)
    assert torch.all(out["np1_motion"][0, 0][mask_squashed] == 0.0)
    assert torch.all(out["motion_n_to_np1"][0, 0][mask_squashed] == 0.0)
    # Unmasked motion preserved
    keep_idx = (~mask_squashed)
    assert torch.allclose(out["n_motion"][0, 0][keep_idx],
                          sample["n_motion"][0, 0][keep_idx])

    # Non-motion buffers untouched
    for k in ("n_lr", "n_depth", "n_normals", "np1_lr", "np1_depth", "np1_normals"):
        assert torch.equal(out[k], sample[k]), f"{k} should not be modified by RRM"
    # HR-GT untouched
    for k in ("n_gt", "n_half_gt", "np1_gt"):
        assert torch.equal(out[k], sample[k]), f"{k} should not be modified by RRM"


def test_charbonnier_weighted_matches_unweighted_when_weight_is_one():
    torch.manual_seed(6)
    pred = torch.rand((1, 3, 16, 16))
    target = torch.rand((1, 3, 16, 16))
    weight = torch.ones((1, 1, 16, 16))
    l_unweighted = charbonnier(pred, target)
    l_weighted = charbonnier(pred, target, weight=weight)
    assert abs(l_unweighted.item() - l_weighted.item()) < 1e-6


def test_charbonnier_weighted_2x_in_patches_exceeds_unweighted():
    """2x weight in 25% of the image area should produce a strictly larger
    loss than the unweighted version (since residual is non-zero on average)."""
    torch.manual_seed(7)
    pred = torch.rand((1, 3, 16, 16))
    target = torch.rand((1, 3, 16, 16))
    weight = torch.ones((1, 1, 16, 16))
    weight[..., :8, :8] = 2.0   # 25% area at 2x
    l_unweighted = charbonnier(pred, target)
    l_weighted = charbonnier(pred, target, weight=weight)
    assert l_weighted.item() > l_unweighted.item()


def test_oss_fx_loss_accepts_rrm_weight_and_changes_total():
    """The full oss_fx_loss should accept rrm_weight_main/inter and
    produce different totals than without."""
    torch.manual_seed(8)
    pred = torch.rand((1, 3, 16, 16))
    gt = torch.rand((1, 3, 16, 16))
    rrm = torch.ones((1, 1, 16, 16))
    rrm[..., :8, :8] = 2.0
    _, parts_no = oss_fx_loss(
        out_main=pred, gt_main=gt,
        lambda_charbonnier=1.0, lambda_lpips=0.0,
        lambda_fg=0.0, lambda_fg_lpips=0.0, lambda_temp_consistency=0.0,
    )
    _, parts_yes = oss_fx_loss(
        out_main=pred, gt_main=gt,
        lambda_charbonnier=1.0, lambda_lpips=0.0,
        lambda_fg=0.0, lambda_fg_lpips=0.0, lambda_temp_consistency=0.0,
        rrm_weight_main=rrm,
    )
    assert parts_yes["sr_charbonnier"] > parts_no["sr_charbonnier"], (
        "RRM 2x weighting in a patch should increase the reported SR Charbonnier"
    )
