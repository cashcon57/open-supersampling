"""Tests for ``scripts.sr_train_v6`` training-loop wiring."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

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
