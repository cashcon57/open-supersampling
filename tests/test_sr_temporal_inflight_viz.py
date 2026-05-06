from __future__ import annotations

import torch

from scripts.sr_temporal_inflight_viz import _comparison_panels


def test_comparison_panels_adds_7_panel_v6_gaussian_strip() -> None:
    h, w = 3, 5
    panel = torch.zeros(3, h, w)

    panels, labels = _comparison_panels(
        lr_up=panel,
        bicubic=panel,
        baseline=panel,
        pixel=panel,
        gaussian=panel,
        v6=panel,
        gt=panel,
        err_rgb=panel,
    )
    strip = torch.cat(panels, dim=-1)

    assert strip.shape == (3, h, w * 7)
    assert labels == [
        "LR-bilinear",
        "bicubic",
        "v5-pixel-temporal",
        "v5-Gaussian",
        "v6",
        "GT",
        "|err| heatmap",
    ]


def test_comparison_panels_keeps_legacy_v5_only_strip() -> None:
    h, w = 3, 5
    panel = torch.zeros(3, h, w)

    panels, labels = _comparison_panels(
        lr_up=panel,
        bicubic=panel,
        baseline=panel,
        pixel=panel,
        gt=panel,
        err_rgb=panel,
    )
    strip = torch.cat(panels, dim=-1)

    assert strip.shape == (3, h, w * 6)
    assert labels == [
        "LR-bilinear",
        "bicubic",
        "v4-baseline",
        "v5-temporal",
        "GT",
        "|err| heatmap",
    ]
