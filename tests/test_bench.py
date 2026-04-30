from __future__ import annotations

import numpy as np
import torch

from ors.bench.fsr1_reference import fsr1_upscale
from ors.bench.quality_runner import QualityRunner


def test_fsr1_runs_without_error() -> None:
    lr = np.random.rand(64, 64, 3).astype(np.float32)
    out = fsr1_upscale(lr, scale_factor=2.0)
    assert out.shape == (128, 128, 3)
    assert np.isfinite(out).all()


def test_quality_runner_returns_all_methods() -> None:
    lr = torch.rand(3, 64, 64)
    gt = torch.rand(3, 128, 128)
    result = QualityRunner(scale_factor=2.0, device="cpu").run_methods(lr, gt)
    assert set(result) == {"bilinear", "bicubic", "fsr1"}
