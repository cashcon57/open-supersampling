"""Unit tests for scripts/sr_eval_v7.py (v7 Phase 3 eval script).

These tests synthesize:
  - A tiny V7Model checkpoint (no TartanAir on disk).
  - A fake triplet dataset matching the
    TartanAirIntermediateTriplets[idx] return shape.

and run the eval entry point end-to-end, then parse the emitted JSON to
verify the contract from the Phase 3 plan.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest
import torch

# Make the scripts/ directory importable so we can pull in sr_eval_v7
# as a regular module without subprocess.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPTS_DIR = _REPO_ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import sr_eval_v7  # noqa: E402
from oss.sr.v7.model import V7Config, V7Model  # noqa: E402


# ---------------------------------------------------------------------
# Fixtures: tiny pico-shape model + synthetic triplets
# ---------------------------------------------------------------------


_LR_H, _LR_W = 8, 12
_HR_H, _HR_W = _LR_H * 2, _LR_W * 2


def _tiny_cfg() -> V7Config:
    return V7Config(
        in_channels=9, scale=2, feat_dim=8, latent_rank=4,
        canvas_capacity=512, backbone_blocks=1,
        backbone_kind="placeholder",
        enable_spawner=True, spawner_k_per_tile=2, spawner_tile_size=8,
    )


def _make_synthetic_triplet(seed: int) -> dict:
    """Build one triplet dict in the same shape that
    TartanAirIntermediateTriplets.__getitem__ produces."""
    g = torch.Generator().manual_seed(seed)

    def _rand_lr():
        return {
            "lr": torch.rand((3, _LR_H, _LR_W), generator=g),
            "depth": torch.rand((1, _LR_H, _LR_W), generator=g),
            "motion": torch.rand((2, _LR_H, _LR_W), generator=g),
            "normals": torch.rand((3, _LR_H, _LR_W), generator=g),
            "gt_hr": torch.rand((3, _HR_H, _HR_W), generator=g),
        }

    n = _rand_lr()
    np1 = _rand_lr()
    n_half_gt = torch.rand((3, _HR_H, _HR_W), generator=g)
    return {
        "n": n,
        "n_half": {"gt_hr": n_half_gt},
        "n_plus_1": np1,
        "motion_n_to_np1": torch.rand((2, _LR_H, _LR_W), generator=g),
    }


class _FakeTripletDataset:
    """Indexable + lengthed dataset of synthetic triplets."""

    def __init__(self, n: int, base_seed: int = 0):
        self._samples = [_make_synthetic_triplet(base_seed + i) for i in range(n)]

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict:
        return self._samples[idx]


def _write_tiny_checkpoint(tmp_path: Path, step: int = 1234) -> Path:
    cfg = _tiny_cfg()
    model = V7Model(cfg)
    model.allocate_canvas("cpu")
    ckpt_path = tmp_path / f"step-{step:08d}.pt"
    torch.save({
        "step": step,
        "model_state": model.state_dict(),
        "cfg": vars(cfg),
        "args": {},
    }, ckpt_path)
    return ckpt_path


# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------


def test_eval_v7_end_to_end_emits_expected_json(tmp_path):
    """Build a tiny checkpoint, run the eval entry point with a fake
    dataset, parse the JSON, and assert that the keys from the Phase
    3 plan are present and finite."""
    ckpt_path = _write_tiny_checkpoint(tmp_path, step=1234)
    output_dir = tmp_path / "eval_out"
    dataset = _FakeTripletDataset(n=3)

    json_path = sr_eval_v7.run_eval_with_dataset(
        checkpoint=ckpt_path,
        dataset=dataset,
        output_dir=output_dir,
        device="cpu",
    )

    assert json_path.exists()
    assert json_path.name == "eval-step-00001234.json"

    with open(json_path) as f:
        data = json.load(f)

    # Top-level keys per Phase 3 plan
    expected_top = {
        "checkpoint", "step", "n_triplets",
        "alpha_1_sr", "alpha_0_5_oss_fx",
        "alpha_0_5_bicubic_baseline",
        "delta_oss_fx_over_bicubic_psnr_db",
        "canvas_health_final",
    }
    assert expected_top.issubset(set(data.keys())), (
        f"missing keys: {expected_top - set(data.keys())}"
    )

    assert data["step"] == 1234
    assert data["n_triplets"] == 3

    # Each metric bundle has psnr + ssim + lpips keys.
    for bundle_key in ("alpha_1_sr", "alpha_0_5_oss_fx", "alpha_0_5_bicubic_baseline"):
        bundle = data[bundle_key]
        assert set(bundle.keys()) == {"psnr", "ssim", "lpips"}, (
            f"{bundle_key} unexpected keys: {bundle.keys()}"
        )
        # PSNR must always be present and finite.
        assert isinstance(bundle["psnr"], (int, float))
        assert math.isfinite(bundle["psnr"])

    # delta must equal alpha_0_5_oss_fx - alpha_0_5_bicubic_baseline (PSNR).
    expected_delta = (
        data["alpha_0_5_oss_fx"]["psnr"]
        - data["alpha_0_5_bicubic_baseline"]["psnr"]
    )
    assert math.isclose(
        data["delta_oss_fx_over_bicubic_psnr_db"], expected_delta, rel_tol=1e-9, abs_tol=1e-9
    )

    # Canvas-health shape sanity (spawner is enabled).
    health = data["canvas_health_final"]
    assert {"count", "mean_opacity", "mean_L_diag"}.issubset(health.keys())
    assert health["count"] > 0   # spawner fires during forward(spawn_at_t=...)


def test_bicubic_midpoint_baseline_psnr_against_known_input():
    """Direct test of the bicubic-midpoint helper + PSNR formula.

    Construct an LR tensor whose bicubic upsample exactly matches a
    constructed GT. PSNR should be high (>= ~60 dB given bicubic is
    not an identity but for a uniform image it nearly is).
    """
    torch.manual_seed(0)
    # A flat-grey LR (3 channels, all 0.5) bicubic-upsampled is also
    # flat-grey 0.5. Score against a flat-grey GT of 0.5 -> MSE = 0
    # -> PSNR clipped to 99.0 by our helper.
    lr = torch.full((1, 9, _LR_H, _LR_W), 0.5)
    gt = torch.full((1, 3, _HR_H, _HR_W), 0.5)
    bi = sr_eval_v7._bicubic_midpoint(lr, (_HR_H, _HR_W))
    assert bi.shape == (1, 3, _HR_H, _HR_W)
    # bicubic of constant input is the same constant (within fp tol).
    assert torch.allclose(bi, gt, atol=1e-5)

    psnr = sr_eval_v7._psnr(bi, gt)
    assert psnr >= 60.0  # all-zero MSE -> capped at 99.0; any tiny noise still > 60

    # Now a noisy case: random GT, bicubic of (different) random LR.
    # We just sanity-check that PSNR is finite and follows the formula.
    lr2 = torch.rand((1, 9, _LR_H, _LR_W))
    gt2 = torch.rand((1, 3, _HR_H, _HR_W))
    bi2 = sr_eval_v7._bicubic_midpoint(lr2, (_HR_H, _HR_W))
    mse = float(((bi2.clamp(0, 1) - gt2.clamp(0, 1)) ** 2).mean().item())
    expected_psnr = 20.0 * math.log10(1.0 / math.sqrt(max(mse, 1e-12)))
    assert math.isclose(
        sr_eval_v7._psnr(bi2, gt2), expected_psnr, rel_tol=1e-6, abs_tol=1e-6
    )


def test_eval_v7_psnr_matches_phase3_plan_formula():
    """Independent verification that _psnr uses the
    20*log10(1/sqrt(mse)) formula prescribed by the Phase 3 plan
    (not, e.g., 10*log10(1/mse) with a different data_range)."""
    pred = torch.zeros((1, 3, 4, 4))
    gt = torch.full((1, 3, 4, 4), 0.5)
    # MSE = 0.25, sqrt = 0.5, 1/0.5 = 2, log10(2) = 0.30103, *20 = 6.0206
    psnr = sr_eval_v7._psnr(pred, gt)
    assert math.isclose(psnr, 20.0 * math.log10(2.0), rel_tol=1e-6, abs_tol=1e-6)
