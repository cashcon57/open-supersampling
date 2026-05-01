"""Smoke test: render one tiny image pair from Mitsuba's bundled cbox scene."""
import numpy as np
import pytest
from oss.render.mitsuba_pipeline import render_pair


def test_render_pair_shapes():
    result = render_pair(
        scene_name="cbox", view_index=0,
        spp_noisy=1, spp_gt=64, resolution=(64, 64),
    )
    assert set(result.keys()) >= {
        "noisy", "ground_truth", "albedo", "normal", "depth", "motion"
    }
    assert result["noisy"].shape == (64, 64, 3)
    assert result["ground_truth"].shape == (64, 64, 3)
    assert result["albedo"].shape == (64, 64, 3)
    assert result["normal"].shape == (64, 64, 3)
    assert result["depth"].shape == (64, 64, 1)
    assert result["motion"].shape == (64, 64, 2)
    assert result["noisy"].dtype == np.float32
