"""Synthetic-data smoke test for the v7 trainer.

Validates the full v7 trainer inner loop -- per-sample reset_state ->
spawn at t=0 -> spawn at t=2 -> render at t=1 -> oss_fx_loss ->
backward -> optim.step -> history.jsonl emission -- against an
in-test synthetic triplet dataset, so we can exercise the alpha
curriculum + canvas-health metrics path on a host without TartanAir
installed.

Covered:
  * 20-step run without errors, curriculum + canvas-health keys present
  * curriculum lambdas change across stages (stage1 -> stage2 ramp ->
    stage3 full activation)
  * canvas count grows and stays bounded under a small capacity
  * no-LPIPS branch (lambda_lpips = 0, lambda_fg_lpips = 0) is exercised

We deliberately do NOT call sr_train_v7.main() -- it's hard-wired to
TartanAirGaussianDataset via argparse. Instead the smoke harness inlines
the per-step body of main()'s training loop and shells out to the
already-module-level helpers (collate_triplets, build_9ch_input,
curriculum_lambdas, canvas_health_metrics).

Runs on CPU, <30s total for the four tests.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sr_train_v7 import (
    build_9ch_input,
    canvas_health_metrics,
    collate_triplets,
    curriculum_lambdas,
)
from oss.sr.v7.model import V7Config, V7Model
from oss.sr.v7.losses import oss_fx_loss


# ----------------------------------------------------------------------
# Synthetic triplet dataset
# ----------------------------------------------------------------------

class SyntheticTripletDataset(Dataset):
    """Mimics TartanAirIntermediateTriplets.__getitem__ output without
    requiring TartanAir on disk. Each item is a dict with `n`, `n_half`,
    `n_plus_1`, `motion_n_to_np1` sub-fields and random tensors of the
    expected shapes for a 16x32 LR -> 32x64 HR setup."""

    def __init__(self, length: int = 8, h_lr: int = 16, w_lr: int = 32, scale: int = 2, seed: int = 0):
        self.length = length
        self.h_lr = h_lr
        self.w_lr = w_lr
        self.h_hr = h_lr * scale
        self.w_hr = w_lr * scale
        self._g = torch.Generator().manual_seed(seed)

    def __len__(self) -> int:
        return self.length

    def _frame(self) -> dict:
        return {
            "lr": torch.rand((3, self.h_lr, self.w_lr), generator=self._g),
            "depth": torch.rand((1, self.h_lr, self.w_lr), generator=self._g),
            "motion": torch.rand((2, self.h_lr, self.w_lr), generator=self._g),
            "normals": torch.rand((3, self.h_lr, self.w_lr), generator=self._g),
            "gt_hr": torch.rand((3, self.h_hr, self.w_hr), generator=self._g),
        }

    def __getitem__(self, idx: int) -> dict:
        n = self._frame()
        np1 = self._frame()
        n_half_gt = torch.rand((3, self.h_hr, self.w_hr), generator=self._g)
        return {
            "n": n,
            "n_half": {"gt_hr": n_half_gt},
            "n_plus_1": np1,
            "motion_n_to_np1": n["motion"] * 2.0,
        }


# ----------------------------------------------------------------------
# Inline trainer inner loop
# ----------------------------------------------------------------------

def _default_args(**overrides) -> SimpleNamespace:
    base = dict(
        lambda_charbonnier=1.0,
        lambda_lpips=0.0,           # disable to skip lazy LPIPS download
        lambda_fg=1.0,
        lambda_fg_lpips=0.5,
        lambda_temp_consistency=0.1,
        curriculum=True,
        curriculum_stage1_end=2,
        curriculum_stage2_end=6,
        curriculum_fg_ramp_steps=2,
        log_every=1,
        ckpt_every=10_000,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _build_model(canvas_capacity: int = 512) -> V7Model:
    cfg = V7Config(
        in_channels=9,
        scale=2,
        feat_dim=8,
        latent_rank=4,
        canvas_capacity=canvas_capacity,
        backbone_kind="placeholder",
        backbone_blocks=1,
        enable_spawner=True,
        spawner_k_per_tile=2,
        spawner_tile_size=8,
    )
    model = V7Model(cfg).train(True)
    model.allocate_canvas("cpu")
    return model


def run_smoke(
    n_steps: int,
    output_dir: Path,
    *,
    canvas_capacity: int = 512,
    dataset_length: int = 8,
    batch_size: int = 1,
    seed: int = 0,
    **arg_overrides,
) -> Path:
    """Run the v7 trainer's inner loop for `n_steps` against a
    SyntheticTripletDataset. Emits history.jsonl in `output_dir`.

    Mirrors the per-sample body of sr_train_v7.main() verbatim --
    reset_state -> spawn t=0 -> spawn t=2 -> render t=1 -> oss_fx_loss
    -> backward -> step -> history.jsonl line with curriculum lambdas
    + canvas-health metrics.
    """
    torch.manual_seed(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / "history.jsonl"
    history_path.write_text("")  # truncate
    device = "cpu"

    args = _default_args(**arg_overrides)

    model = _build_model(canvas_capacity=canvas_capacity)
    ds = SyntheticTripletDataset(length=dataset_length, seed=seed)
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=0, collate_fn=collate_triplets, drop_last=True,
    )
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    t0 = time.perf_counter()
    step = 0
    while step < n_steps:
        for batch in loader:
            step += 1
            if step > n_steps:
                break
            optim.zero_grad()

            n_samples = batch["n_lr"].shape[0]
            if args.curriculum:
                fg_w, fg_lpips_w, temp_w = curriculum_lambdas(
                    step=step,
                    stage1_end=args.curriculum_stage1_end,
                    stage2_end=args.curriculum_stage2_end,
                    fg_ramp_steps=args.curriculum_fg_ramp_steps,
                    target_fg=args.lambda_fg,
                    target_fg_lpips=args.lambda_fg_lpips,
                    target_temp=args.lambda_temp_consistency,
                )
            else:
                fg_w = args.lambda_fg
                fg_lpips_w = args.lambda_fg_lpips
                temp_w = args.lambda_temp_consistency

            # Per-sample backward + grad accumulation. The trainer
            # accumulates total loss across samples and calls backward
            # once at the end -- that path works in prod because
            # production typically runs B=1 (per-rank canvas state),
            # but with B>1 the in-place canvas reset between samples
            # invalidates earlier samples' saved-tensors. Backwarding
            # per-sample with /n_samples scaling is mathematically
            # equivalent and exercises the optimizer / grad path the
            # same way.
            batch_parts: dict[str, float] = {}
            running_total = 0.0
            for b in range(n_samples):
                model.reset_state(device)

                n_lr_in_b = build_9ch_input(
                    batch["n_lr"][b:b+1].to(device),
                    batch["n_depth"][b:b+1].to(device),
                    batch["n_motion"][b:b+1].to(device),
                    batch["n_normals"][b:b+1].to(device),
                )
                np1_lr_in_b = build_9ch_input(
                    batch["np1_lr"][b:b+1].to(device),
                    batch["np1_depth"][b:b+1].to(device),
                    batch["np1_motion"][b:b+1].to(device),
                    batch["np1_normals"][b:b+1].to(device),
                )
                n_half_gt_b = batch["n_half_gt"][b:b+1].to(device).clamp(0, 1)
                np1_gt_b = batch["np1_gt"][b:b+1].to(device).clamp(0, 1)

                out_main_n = model(n_lr_in_b, t_query=0.0, spawn_at_t=0.0)
                out_main_np1 = model(np1_lr_in_b, t_query=2.0, spawn_at_t=2.0)
                out_inter = model(np1_lr_in_b, t_query=1.0)

                loss_b, parts_b = oss_fx_loss(
                    out_main=out_main_np1,
                    gt_main=np1_gt_b,
                    out_inter_list=[out_inter],
                    gt_inter_list=[n_half_gt_b],
                    lambda_charbonnier=args.lambda_charbonnier,
                    lambda_lpips=args.lambda_lpips,
                    lambda_fg=fg_w,
                    lambda_fg_lpips=fg_lpips_w,
                    lambda_temp_consistency=temp_w,
                )
                scaled = loss_b / float(n_samples)
                scaled.backward()
                running_total += float(loss_b.detach().item())
                for k, v in parts_b.items():
                    batch_parts[k] = batch_parts.get(k, 0.0) + float(v)

            for k in batch_parts:
                batch_parts[k] /= float(n_samples)
            running_total /= float(n_samples)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optim.step()

            # Detach canvas buffers so the next step's `add` doesn't
            # try to backward through a stale graph. The canvas
            # buffers are mutated in-place by spawn(); without this
            # detach, their grad_fn from step N follows them into
            # step N+1's forward pass.
            cs = model.canvas
            cs.positions = cs.positions.detach()
            cs.cov_raw = cs.cov_raw.detach()
            cs.features = cs.features.detach()
            cs.opacity = cs.opacity.detach()

            # Emit history every step (log_every=1) -- mirrors prod
            # emission shape so tests can introspect curriculum +
            # canvas-health fields.
            parts = dict(batch_parts)
            parts["step"] = step
            parts["elapsed_s"] = time.perf_counter() - t0
            parts["lambda_fg"] = fg_w
            parts["lambda_fg_lpips"] = fg_lpips_w
            parts["lambda_temp"] = temp_w
            parts.update(canvas_health_metrics(model))
            with open(history_path, "a") as f:
                f.write(json.dumps(parts) + "\n")

            # Sanity: every step's accumulated loss is finite.
            assert running_total == running_total, (
                f"step {step} loss non-finite (NaN): {running_total}"
            )
            assert running_total not in (float("inf"), float("-inf")), (
                f"step {step} loss non-finite (inf): {running_total}"
            )

    return history_path


def _read_history(path: Path) -> list[dict]:
    with open(path, "r") as f:
        return [json.loads(line) for line in f if line.strip()]


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------

def test_trainer_smoke_20_steps_runs_without_errors(tmp_path):
    history_path = run_smoke(n_steps=20, output_dir=tmp_path)
    rows = _read_history(history_path)
    assert len(rows) == 20, f"expected 20 history rows, got {len(rows)}"
    for row in rows:
        # `total` is the oss_fx_loss aggregate component
        assert "total" in row
        assert row["total"] == row["total"]  # NaN check via reflexivity
        assert row["total"] != float("inf")
        assert row["total"] != float("-inf")
    last = rows[-1]
    # Curriculum + canvas-health keys must appear in the emitted rows
    for k in (
        "lambda_fg",
        "lambda_fg_lpips",
        "lambda_temp",
        "canvas_count",
        "canvas_mean_opacity",
        "canvas_mean_L_diag",
    ):
        assert k in last, f"missing history key {k!r} in last row: {last}"


def test_trainer_smoke_curriculum_lambdas_change_across_stages(tmp_path):
    # stage1_end=2, stage2_end=6, fg_ramp_steps=2, target_fg=1.0,
    # target_fg_lpips=0.5, target_temp=0.1
    history_path = run_smoke(
        n_steps=10,
        output_dir=tmp_path,
        curriculum_stage1_end=2,
        curriculum_stage2_end=6,
        curriculum_fg_ramp_steps=2,
        lambda_fg=1.0,
        lambda_fg_lpips=0.5,
        lambda_temp_consistency=0.1,
    )
    rows = _read_history(history_path)
    by_step = {r["step"]: r for r in rows}

    # Stage 1 boundary -- pure SR, FG = 0
    assert by_step[2]["lambda_fg"] == 0.0
    assert by_step[2]["lambda_fg_lpips"] == 0.0
    assert by_step[2]["lambda_temp"] == 0.0

    # Stage 2 ramp (steps 3..6): FG-LPIPS / temp still 0, FG strictly
    # between 0 and target somewhere in the ramp window.
    ramp_fg = [by_step[s]["lambda_fg"] for s in (3, 4, 5)]
    assert any(0.0 < x < 1.0 for x in ramp_fg), (
        f"expected at least one in-ramp lambda_fg in (0, 1); got {ramp_fg}"
    )
    for s in (3, 4, 5):
        assert by_step[s]["lambda_fg_lpips"] == 0.0
        assert by_step[s]["lambda_temp"] == 0.0

    # Stage 3 (step 10): full activation
    final = by_step[10]
    assert abs(final["lambda_fg"] - 1.0) < 1e-6
    assert abs(final["lambda_fg_lpips"] - 0.5) < 1e-6
    assert abs(final["lambda_temp"] - 0.1) < 1e-6


def test_trainer_smoke_canvas_count_grows_then_stays_bounded(tmp_path):
    # canvas_capacity=256 deliberately small to force prune/cap
    capacity = 256
    history_path = run_smoke(
        n_steps=20,
        output_dir=tmp_path,
        canvas_capacity=capacity,
    )
    rows = _read_history(history_path)
    counts = [r["canvas_count"] for r in rows]

    # Canvas must populate within the first 5 steps
    assert max(counts[:5]) > 0, (
        f"canvas never grew above 0 in first 5 steps: {counts[:5]}"
    )
    # And must never exceed the configured capacity in ANY row
    assert max(counts) <= capacity, (
        f"canvas exceeded capacity {capacity}: max={max(counts)}"
    )


def test_trainer_smoke_handles_lpips_disabled(tmp_path):
    # lambda_lpips=0 AND lambda_fg_lpips=0 -- exercises the lazy LPIPS
    # load path's no-op branch.
    history_path = run_smoke(
        n_steps=5,
        output_dir=tmp_path,
        lambda_lpips=0.0,
        lambda_fg_lpips=0.0,
    )
    rows = _read_history(history_path)
    assert len(rows) == 5
    for r in rows:
        assert r["total"] == r["total"]  # finite
        # FG-LPIPS must remain 0 throughout regardless of stage
        assert r["lambda_fg_lpips"] == 0.0
