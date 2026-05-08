"""Tests for ``scripts.sr_train_v6`` training-loop wiring."""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
from pathlib import Path
import socket
import subprocess
import sys
import textwrap
import warnings

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from types import SimpleNamespace

import scripts.sr_train_v6 as train_v6
from oss.sr.v6.ema import EMAModel
import oss.sr.v6.losses as losses_mod


class CheapCompositeLoss(nn.Module):
    def __init__(self, gan_warmup_until_step: int = 20_000):
        super().__init__()
        self.gan_warmup_until_step = int(gan_warmup_until_step)

    def forward(
        self,
        pred,
        target,
        fake_logits,
        step,
        pred_prev=None,
        motion_lr=None,
        **_kwargs,
    ):
        char = (pred - target).abs().mean()
        vgg = pred.square().mean() * 0.01
        lpips = (pred - target).square().mean() * 0.01
        wav = char * 0.1
        sobel = char * 0.2
        gan = -fake_logits.mean() if fake_logits is not None and step >= self.gan_warmup_until_step else pred.new_zeros(())
        temporal = (
            (pred - pred_prev).abs().mean()
            if pred_prev is not None and motion_lr is not None
            else pred.new_zeros(())
        )
        total = char + vgg + lpips + wav + sobel + 0.05 * gan + 0.5 * temporal
        return total, {
            "total": float(total.detach()),
            "charbonnier": float(char.detach()),
            "vgg": float(vgg.detach()),
            "lpips": float(lpips.detach()),
            "wavelet": float(wav.detach()),
            "sobel": float(sobel.detach()),
            "gan": float(gan.detach()),
            "temporal": float(temporal.detach()),
        }


class NonFiniteCompositeLoss(nn.Module):
    def forward(self, pred, target, fake_logits, step, **_kwargs):
        loss = pred.sum() * pred.new_tensor(float("nan"))
        return loss, {"total": float("nan")}


class TinyDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 1, 1)

    def forward(self, x):
        return self.conv(x)


class TinyGenerator(nn.Module):
    scale = 2

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(9, 3, 1)
        self.prune_calls = 0
        self._canvas_state = None
        self.reset_count = 0
        self.frame_start_counts = []

    def reset_state(self, device=None):
        self.reset_count += 1
        self._canvas_state = None

    def maybe_prune(self):
        self.prune_calls += 1
        return 0

    def forward(self, lr_inputs, motion_lr=None, frame_index=0):
        self.frame_start_counts.append(
            0 if self._canvas_state is None else int(self._canvas_state.count)
        )
        x = self.conv(lr_inputs)
        out = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        self._canvas_state = SimpleNamespace(
            count=(1 if self._canvas_state is None else int(self._canvas_state.count) + 1)
        )
        return out


@pytest.fixture()
def cheap_trainer(monkeypatch):
    monkeypatch.setattr(train_v6, "V6CompositeLoss", CheapCompositeLoss)
    monkeypatch.setattr(train_v6, "UNetDiscriminator", TinyDiscriminator)


def _metric_rows(output_dir: Path) -> list[dict]:
    return [json.loads(line) for line in (output_dir / "metrics.json").read_text().splitlines()]


def _save_synthetic_v5_teacher_ckpt(path: Path) -> Path:
    from oss.sr.temporal import TemporalSRModel

    model = TemporalSRModel(
        in_channels=12,
        scale=2,
        tier="pico",
        backbone_kind="simple",
        zero_gbuffer_into_backbone=True,
    )
    torch.save(
        {
            "kind": "temporal",
            "args": {
                "in_channels": 12,
                "scale": 2,
                "tier": "pico",
                "backbone_kind": "simple",
                "zero_gbuffer_into_backbone": True,
            },
            "temporal_model": model.state_dict(),
        },
        path,
    )
    return path


def test_smoke_no_bf16_flag_recognized(tmp_path):
    repo = Path(__file__).resolve().parents[3]
    code = textwrap.dedent(
        f"""
        import pathlib
        import scripts.sr_train_v6 as train_v6

        args = train_v6.parse_args([
            "--output-dir", {str(tmp_path)!r},
            "--smoke",
            "--no-bf16",
        ])
        assert args.smoke is True
        assert args.bf16 is False
        """
    )
    res = subprocess.run(
        [sys.executable, "-c", code],
        cwd=repo,
        env={**os.environ, "PYTHONPATH": str(repo)},
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert res.returncode == 0, res.stdout + res.stderr


def test_spawn_subpixel_jitter_flag_recognized(tmp_path):
    args = train_v6.parse_args([
        "--output-dir", str(tmp_path),
        "--smoke",
        "--spawn-subpixel-jitter",
    ])
    args = train_v6.normalize_args(args)

    assert args.spawn_subpixel_jitter is True
    assert args.spawn_offset_random is False


def test_smoke_ddp_singleprocess_path(tmp_path, cheap_trainer, monkeypatch):
    for name in ("WORLD_SIZE", "RANK", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"):
        monkeypatch.delenv(name, raising=False)

    rc = train_v6.main([
        "--output-dir", str(tmp_path),
        "--smoke",
        "--device", "cpu",
        "--backbone", "hat-tiny",
        "--patch-size", "32",
        "--batch-size", "1",
        "--grad-accum", "1",
        "--max-steps", "1",
    ])

    assert rc == 0
    assert _metric_rows(tmp_path)[0]["step"] == 1


def test_smoke_runs_five_steps_checkpoints_and_metrics(tmp_path, cheap_trainer):
    rc = train_v6.main([
        "--output-dir", str(tmp_path),
        "--smoke",
        "--device", "cpu",
        "--backbone", "hat-tiny",
        "--patch-size", "32",
        "--batch-size", "1",
        "--grad-accum", "1",
        "--ckpt-every", "2",
    ])

    assert rc == 0
    rows = _metric_rows(tmp_path)
    assert [r["step"] for r in rows] == [1, 2, 3, 4, 5]
    for row in rows:
        for key in (
            "step", "loss_total", "loss_charbonnier", "loss_lpips",
            "loss_msvgg", "loss_wavelet", "loss_sobel", "loss_gan_g",
            "loss_gan_d", "loss_tc", "loss_v5_kd", "lambda_v5_kd",
            "lr_g", "lr_d", "ema_decay",
        ):
            assert key in row
    assert (tmp_path / "step-00000002.pt").exists()
    assert (tmp_path / "step-00000004.pt").exists()
    assert (tmp_path / "step-00000005.pt").exists()
    assert any(row["loss_tc"] > 0.0 for row in rows)


def test_early_checkpoint_step(tmp_path, cheap_trainer):
    rc = train_v6.main([
        "--output-dir", str(tmp_path),
        "--smoke",
        "--device", "cpu",
        "--backbone", "hat-tiny",
        "--patch-size", "32",
        "--batch-size", "1",
        "--grad-accum", "1",
        "--first-ckpt-step", "2",
        "--ckpt-every", "100",
    ])

    assert rc == 0
    assert (tmp_path / "step-00000002.pt").exists()


def test_smoke_v5_teacher_kd_decays_to_zero(tmp_path, cheap_trainer):
    ckpt = _save_synthetic_v5_teacher_ckpt(tmp_path / "v5-teacher.pt")

    rc = train_v6.main([
        "--output-dir", str(tmp_path / "run"),
        "--smoke",
        "--device", "cpu",
        "--backbone", "hat-tiny",
        "--patch-size", "32",
        "--batch-size", "1",
        "--grad-accum", "1",
        "--max-steps", "2",
        "--v5-teacher-ckpt", str(ckpt),
        "--v5-teacher-decay-end", "2",
    ])

    assert rc == 0
    rows = _metric_rows(tmp_path / "run")
    assert rows[0]["step"] == 1
    assert rows[0]["lambda_v5_kd"] > 0.0
    assert rows[0]["loss_v5_kd"] > 0.0
    assert rows[1]["step"] == 2
    assert rows[1]["lambda_v5_kd"] == 0.0
    assert rows[1]["loss_v5_kd"] == 0.0


def test_missing_v5_teacher_ckpt_warns_and_disables(tmp_path, caplog):
    missing = tmp_path / "missing-v5.pt"

    with caplog.at_level(logging.WARNING, logger="oss.sr.v6.train"):
        teacher = train_v6.load_v5_teacher(missing, device="cpu")

    assert teacher is None
    assert "v5 teacher checkpoint not found" in caplog.text


def _loopback_bind_available() -> tuple[bool, str]:
    sock = socket.socket()
    try:
        sock.bind(("127.0.0.1", 0))
        return True, str(sock.getsockname()[1])
    except OSError as exc:
        return False, str(exc)
    finally:
        sock.close()


def test_smoke_ddp_multirank_probe_or_warn(tmp_path):
    bind_ok, port_or_error = _loopback_bind_available()
    if not torch.cuda.is_available() and not bind_ok:
        warnings.warn(
            f"skipping DDP smoke subprocess: no CUDA and loopback bind blocked: {port_or_error}",
            RuntimeWarning,
        )
        return
    if not bind_ok:
        warnings.warn(
            f"skipping DDP smoke subprocess: loopback bind blocked: {port_or_error}",
            RuntimeWarning,
        )
        return

    repo = Path(__file__).resolve().parents[3]
    probe_script = tmp_path / "ddp_probe.py"
    probe_script.write_text(
        textwrap.dedent(
            """
            import argparse
            import os

            import torch
            import torch.distributed as dist
            from torch.nn.parallel import DistributedDataParallel as DDP

            import scripts.sr_train_v6 as train_v6
            from oss.sr.v6.model import V6Config, V6Model

            dist.init_process_group("gloo")
            rank = int(os.environ["RANK"])
            world_size = int(os.environ["WORLD_SIZE"])
            model = V6Model(V6Config(backbone="hat-tiny")).cpu()
            ddp = DDP(model, find_unused_parameters=True)
            args = argparse.Namespace(device="cpu", bf16=False)
            train_v6.run_ddp_smoke_probe(
                ddp,
                args=args,
                rank=rank,
                world_size=world_size,
            )
            dist.destroy_process_group()
            """
        ),
        encoding="utf-8",
    )
    res = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nproc_per_node=2",
            "--master_addr=127.0.0.1",
            f"--master_port={port_or_error}",
            str(probe_script),
        ],
        cwd=repo,
        env={**os.environ, "PYTHONPATH": str(repo)},
        text=True,
        capture_output=True,
        timeout=60,
    )
    combined = res.stdout + res.stderr
    if res.returncode != 0 and any(
        token in combined
        for token in ("EPERM", "Operation not permitted", "failed to bind", "Rendezvous")
    ):
        warnings.warn(f"DDP smoke subprocess skipped after launcher bind failure: {combined}")
        return
    assert res.returncode == 0, combined


def test_auto_resume_continues_at_next_step(tmp_path, cheap_trainer):
    first = train_v6.main([
        "--output-dir", str(tmp_path),
        "--smoke",
        "--device", "cpu",
        "--backbone", "hat-tiny",
        "--patch-size", "32",
        "--batch-size", "1",
        "--grad-accum", "1",
        "--max-steps", "3",
        "--ckpt-every", "1",
    ])
    assert first == 0

    second = train_v6.main([
        "--output-dir", str(tmp_path),
        "--smoke",
        "--device", "cpu",
        "--backbone", "hat-tiny",
        "--patch-size", "32",
        "--batch-size", "1",
        "--grad-accum", "1",
        "--max-steps", "5",
        "--ckpt-every", "1",
    ])
    assert second == 0

    rows = _metric_rows(tmp_path)
    assert [r["step"] for r in rows] == [1, 2, 3, 4, 5]
    assert (tmp_path / "step-00000003.pt").exists()
    assert (tmp_path / "step-00000005.pt").exists()
    assert any(row["loss_tc"] > 0.0 for row in rows[3:])


def test_load_rng_state_casts_legacy_non_byte_torch_state(caplog):
    torch.manual_seed(1234)
    legacy_state = train_v6._rng_state()
    legacy_state["torch"] = legacy_state["torch"].to(torch.int32)

    torch.manual_seed(9999)
    with caplog.at_level(logging.WARNING, logger="oss.sr.v6.train"):
        train_v6._load_rng_state(legacy_state)

    assert "casting torch RNG state from torch.int32 to torch.uint8" in caplog.text
    assert [record.levelno for record in caplog.records] == [logging.WARNING]


def test_rng_state_save_load_restores_exact_rng_state():
    random.seed(123)
    np.random.seed(456)
    torch.manual_seed(789)

    saved = train_v6._rng_state()
    expected_torch = torch.get_rng_state().clone()
    expected_python = random.getstate()
    expected_numpy = np.random.get_state()

    assert saved["torch"].dtype == torch.uint8
    assert all(state.dtype == torch.uint8 for state in saved["cuda"])

    _ = random.random()
    _ = np.random.random()
    _ = torch.rand(4)

    train_v6._load_rng_state(saved)

    loaded_numpy = np.random.get_state()
    assert torch.equal(torch.get_rng_state(), expected_torch)
    assert random.getstate() == expected_python
    assert loaded_numpy[0] == expected_numpy[0]
    assert np.array_equal(loaded_numpy[1], expected_numpy[1])
    assert loaded_numpy[2:] == expected_numpy[2:]


def test_canvas_continues_inside_trajectory():
    g = TinyGenerator()
    d = TinyDiscriminator()
    loss = CheapCompositeLoss()
    opt_g = torch.optim.AdamW(g.parameters(), lr=1e-4)
    opt_d = torch.optim.AdamW(d.parameters(), lr=1e-4)
    ema = EMAModel(g, decay=0.999)

    parts = train_v6.train_step(
        g, d, loss, opt_g, opt_d, ema,
        _tiny_trajectory_batch(length=2), step=1, args=_tiny_args(),
    )

    assert parts["canvas_count_at_frame1_start"] > 0.0
    assert g.frame_start_counts[:2] == [0, 1]


def test_nonfinite_train_step_clears_grads_and_skips_update():
    g = TinyGenerator()
    d = TinyDiscriminator()
    opt_g = torch.optim.AdamW(g.parameters(), lr=1e-2)
    opt_d = torch.optim.AdamW(d.parameters(), lr=1e-2)
    ema = EMAModel(g, decay=0.999)

    before = [p.detach().clone() for p in g.parameters()]
    for p in g.parameters():
        p.grad = torch.ones_like(p)

    parts = train_v6.train_step(
        g, d, NonFiniteCompositeLoss(), opt_g, opt_d, ema,
        _tiny_batch(), step=1, args=_tiny_args(),
    )

    assert parts["loss_total"] != parts["loss_total"]
    assert all(p.grad is None for p in g.parameters())
    for old, new in zip(before, g.parameters()):
        assert torch.equal(old, new)


def test_composite_loss_debug_reports_first_nonfinite_component(monkeypatch, caplog):
    class TinyVGGLoss(nn.Module):
        def forward(self, pred, target):
            return pred.new_zeros(())

    monkeypatch.setattr(losses_mod, "MultiScaleVGGLoss", TinyVGGLoss)
    monkeypatch.setattr(
        losses_mod,
        "wavelet_l1_loss",
        lambda pred, target: pred.sum() * pred.new_tensor(float("nan")),
    )

    loss_fn = losses_mod.V6CompositeLoss(use_lpips=False, debug_nan=True)
    pred = torch.rand(1, 3, 32, 32)
    target = torch.rand(1, 3, 32, 32)

    with caplog.at_level("WARNING", logger="oss.sr.v6.losses"):
        loss, parts = loss_fn(pred, target, fake_logits=None, step=7)

    assert not torch.isfinite(loss)
    assert parts["first_non_finite_component"] == "wavelet"
    assert "loss component non-finite: name=wavelet" in caplog.text
    assert "step=7" in caplog.text


def test_canvas_resets_between_trajectories():
    g = TinyGenerator()
    d = TinyDiscriminator()
    loss = CheapCompositeLoss()
    opt_g = torch.optim.AdamW(g.parameters(), lr=1e-4)
    opt_d = torch.optim.AdamW(d.parameters(), lr=1e-4)
    ema = EMAModel(g, decay=0.999)

    for step in (1, 2):
        train_v6.train_step(
            g, d, loss, opt_g, opt_d, ema,
            _tiny_trajectory_batch(length=2), step=step, args=_tiny_args(),
        )

    assert g.reset_count == 2
    assert g.frame_start_counts[:4] == [0, 1, 0, 1]


def _tiny_batch():
    return {
        "lr_inputs": torch.rand(1, 9, 16, 16),
        "target": torch.rand(1, 3, 32, 32),
        "motion": torch.zeros(1, 2, 16, 16),
    }


def _tiny_trajectory_batch(length: int = 2):
    return {
        "lr_inputs": torch.rand(1, length, 9, 16, 16),
        "target": torch.rand(1, length, 3, 32, 32),
        "motion": torch.zeros(1, length, 2, 16, 16),
    }


def _tiny_args(warmup_steps: int = 20_000):
    return argparse.Namespace(device="cpu", bf16=False, warmup_steps=warmup_steps)


def test_gan_warmup_skips_d_at_zero_and_fires_after_warmup():
    g = TinyGenerator()
    d = TinyDiscriminator()
    loss = CheapCompositeLoss(gan_warmup_until_step=20_000)
    opt_g = torch.optim.AdamW(g.parameters(), lr=1e-4)
    opt_d = torch.optim.AdamW(d.parameters(), lr=1e-4)
    ema = EMAModel(g, decay=0.999)

    pre = train_v6.train_step(
        g, d, loss, opt_g, opt_d, ema, _tiny_batch(), step=0, args=_tiny_args(),
    )
    assert pre["d_step"] == 0.0
    assert pre["loss_gan_d"] == 0.0

    post = train_v6.train_step(
        g, d, loss, opt_g, opt_d, ema, _tiny_batch(), step=20_001, args=_tiny_args(),
    )
    assert post["d_step"] == 1.0
    assert post["loss_gan_d"] > 0.0


def test_checkpoint_round_trip_restores_full_state(tmp_path):
    args = argparse.Namespace(
        base_lr=1e-4,
        weight_decay=1e-4,
        T0=10,
        num_restarts=1,
        output_dir=tmp_path,
    )
    g = TinyGenerator()
    d = TinyDiscriminator()
    opt_g = train_v6.build_optimizer(g, args)
    opt_d = train_v6.build_optimizer(d, args)
    sched_g = train_v6.build_scheduler(opt_g, args)
    sched_d = train_v6.build_scheduler(opt_d, args)
    ema = EMAModel(g, decay=0.999)

    parts = train_v6.train_step(
        g, d, CheapCompositeLoss(), opt_g, opt_d, ema,
        _tiny_batch(), step=20_001, args=_tiny_args(),
    )
    assert torch.isfinite(torch.tensor(parts["loss_total"]))
    sched_g.step(7)
    sched_d.step(7)
    torch.manual_seed(123)
    train_v6.save_checkpoint(tmp_path, 7, g, d, opt_g, opt_d, sched_g, sched_d, ema, args)
    expected_rand = torch.rand(4)

    g2 = TinyGenerator()
    d2 = TinyDiscriminator()
    opt_g2 = train_v6.build_optimizer(g2, args)
    opt_d2 = train_v6.build_optimizer(d2, args)
    sched_g2 = train_v6.build_scheduler(opt_g2, args)
    sched_d2 = train_v6.build_scheduler(opt_d2, args)
    ema2 = EMAModel(g2, decay=0.5)
    torch.manual_seed(999)

    step = train_v6.load_latest_checkpoint(
        tmp_path, g2, d2, opt_g2, opt_d2, sched_g2, sched_d2, ema2, "cpu",
    )
    assert step == 7
    for p1, p2 in zip(g.parameters(), g2.parameters()):
        assert torch.equal(p1, p2)
    for p1, p2 in zip(d.parameters(), d2.parameters()):
        assert torch.equal(p1, p2)
    assert opt_g2.state_dict()["state"]
    assert opt_d2.state_dict()["state"]
    assert sched_g2.get_last_lr() == pytest.approx(sched_g.get_last_lr())
    assert sched_d2.get_last_lr() == pytest.approx(sched_d.get_last_lr())
    for name in ema.shadow_params:
        assert torch.equal(ema.shadow_params[name], ema2.shadow_params[name])
    assert torch.equal(torch.rand(4), expected_rand)
