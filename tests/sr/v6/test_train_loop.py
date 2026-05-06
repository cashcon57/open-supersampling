"""Tests for ``scripts.sr_train_v6`` training-loop wiring."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import textwrap
import warnings

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import scripts.sr_train_v6 as train_v6
from oss.sr.v6.ema import EMAModel


class CheapCompositeLoss(nn.Module):
    def __init__(self, gan_warmup_until_step: int = 20_000):
        super().__init__()
        self.gan_warmup_until_step = int(gan_warmup_until_step)

    def forward(self, pred, target, fake_logits, step, **_kwargs):
        char = (pred - target).abs().mean()
        vgg = pred.square().mean() * 0.01
        lpips = (pred - target).square().mean() * 0.01
        wav = char * 0.1
        sobel = char * 0.2
        gan = -fake_logits.mean() if fake_logits is not None and step >= self.gan_warmup_until_step else pred.new_zeros(())
        total = char + vgg + lpips + wav + sobel + 0.05 * gan
        return total, {
            "total": float(total.detach()),
            "charbonnier": float(char.detach()),
            "vgg": float(vgg.detach()),
            "lpips": float(lpips.detach()),
            "wavelet": float(wav.detach()),
            "sobel": float(sobel.detach()),
            "gan": float(gan.detach()),
            "temporal": 0.0,
        }


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

    def reset_state(self, device=None):
        return None

    def maybe_prune(self):
        self.prune_calls += 1
        return 0

    def forward(self, lr_inputs, motion_lr=None, frame_index=0):
        x = self.conv(lr_inputs)
        return F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)


@pytest.fixture()
def cheap_trainer(monkeypatch):
    monkeypatch.setattr(train_v6, "V6CompositeLoss", CheapCompositeLoss)
    monkeypatch.setattr(train_v6, "UNetDiscriminator", TinyDiscriminator)


def _metric_rows(output_dir: Path) -> list[dict]:
    return [json.loads(line) for line in (output_dir / "metrics.json").read_text().splitlines()]


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
            "loss_gan_d", "loss_tc", "lr_g", "lr_d", "ema_decay",
        ):
            assert key in row
    assert (tmp_path / "step-00000002.pt").exists()
    assert (tmp_path / "step-00000004.pt").exists()
    assert (tmp_path / "step-00000005.pt").exists()


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


def _tiny_batch():
    return {
        "lr_inputs": torch.rand(1, 9, 16, 16),
        "target": torch.rand(1, 3, 32, 32),
        "motion": torch.zeros(1, 2, 16, 16),
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
