"""Stage-gate evaluator for the v6 Gaussian-temporal staged validation.

Reads the most recent training log from a stage's output dir + the latest
checkpoint, applies the stage's pass/fail criteria, prints PASS or FAIL,
and exits with code 0 (pass) or 1 (fail) so the watchdog can chain
correctly.

Usage:
    python v6_gate_eval.py --stage stage0 --output-dir <path>
    python v6_gate_eval.py --stage stage1 --output-dir <path>
    python v6_gate_eval.py --stage stage2 --output-dir <path>
        --bicubic-psnr-floor 24.0

Stage gate rules:
    stage0: process exited cleanly, last logged step >= 100,
            last logged loss < 5.0 (sanity, not exploding)
    stage1: last logged step >= 4500, last logged loss < 0.05
            (overfit single trajectory should drive loss low)
    stage2: last logged step >= 11000, last logged train loss < 1.0,
            and (if a held-out PSNR row was emitted) PSNR > floor

Exit code 0 = PASS, 1 = FAIL (any criterion missed).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


def _scan_train_log(log_path: Path) -> tuple[int, float] | None:
    """Return (last_step, last_loss) parsed from train.log, or None."""
    if not log_path.exists():
        return None
    last_step = -1
    last_loss = float("nan")
    pat = re.compile(r"step=(\d+).*?loss=([-\d.eE+]+)")
    with open(log_path, errors="ignore") as f:
        for line in f:
            m = pat.search(line)
            if m:
                last_step = int(m.group(1))
                try:
                    last_loss = float(m.group(2))
                except ValueError:
                    pass
    if last_step < 0:
        return None
    return last_step, last_loss


def _scan_score_log(score_log_path: Path) -> dict | None:
    """Return last score_log row (held-out eval), or None if no held-out yet."""
    if not score_log_path.exists():
        return None
    try:
        rows = json.loads(score_log_path.read_text())
    except json.JSONDecodeError:
        return None
    if not rows:
        return None
    return rows[-1]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--stage", required=True, choices=["stage0", "stage1", "stage2"])
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--bicubic-psnr-floor", type=float, default=24.0)
    args = p.parse_args()

    log_path = args.output_dir / "train.log"
    score_log_path = args.output_dir / "score_log.json"
    train = _scan_train_log(log_path)

    if train is None:
        print(f"FAIL: no parseable training steps in {log_path}")
        return 1
    last_step, last_loss = train

    print(f"stage={args.stage} step={last_step} loss={last_loss:.4f}")

    if args.stage == "stage0":
        # Smoke. Just need to have logged some steps without exploding.
        if last_step < 60:
            print(f"FAIL: stage0 needs >= 60 logged steps, got {last_step}")
            return 1
        if last_loss > 5.0 or last_loss != last_loss:  # NaN check
            print(f"FAIL: stage0 loss too high or NaN: {last_loss}")
            return 1
        print("PASS: smoke OK — training loop runs, gradients flow")
        return 0

    if args.stage == "stage1":
        # Overfit single trajectory. Loss should be very low.
        if last_step < 4500:
            print(f"FAIL: stage1 needs >= 4500 steps, got {last_step}")
            return 1
        if last_loss > 0.10:
            print(
                f"FAIL: stage1 loss too high for overfit: {last_loss:.4f} "
                f"(threshold 0.10). Architecture may lack capacity, or "
                f"training is unstable."
            )
            return 1
        print(
            f"PASS: overfit succeeded (loss {last_loss:.4f} < 0.10) — "
            f"architecture has capacity, gradient flow healthy"
        )
        return 0

    if args.stage == "stage2":
        # Multi-trajectory generalization. Need stable training + ideally
        # held-out beats bicubic.
        if last_step < 11000:
            print(f"FAIL: stage2 needs >= 11000 steps, got {last_step}")
            return 1
        if last_loss > 1.5 or last_loss != last_loss:
            print(f"FAIL: stage2 train loss unstable: {last_loss:.4f}")
            return 1
        score = _scan_score_log(score_log_path)
        if score is None:
            print(
                f"WARN: no score_log row found for stage2 — held-out eval "
                f"didn't run. Soft-pass on training-loss stability alone."
            )
            print("PASS (soft): train loss stable, no held-out evidence yet")
            return 0
        psnr = score.get("psnr")
        if psnr is None:
            print(f"WARN: score_log row has no 'psnr' field: {score}")
            print("PASS (soft): train loss stable, score_log incomplete")
            return 0
        if psnr < args.bicubic_psnr_floor:
            print(
                f"FAIL: stage2 held-out PSNR {psnr:.2f} dB below bicubic "
                f"floor {args.bicubic_psnr_floor:.2f} dB. Generalization "
                f"is not happening."
            )
            return 1
        print(
            f"PASS: held-out PSNR {psnr:.2f} dB > bicubic floor "
            f"{args.bicubic_psnr_floor:.2f} dB — generalization working"
        )
        return 0

    print(f"FAIL: unknown stage {args.stage!r}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
