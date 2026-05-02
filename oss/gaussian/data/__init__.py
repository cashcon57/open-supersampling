"""OSS-Gaussian Sprint 4 dataset loaders.

Public API for training-time datasets that feed
``GaussianParamNetwork`` (12-channel input + GT HR target).

Layout:
    base.py         — GaussianTrainingExample, GaussianDataset abstract base
    lr_synthesis.py — EngineAliasedLRSynth + individual pipeline helpers
    sintel.py       — SintelGaussianDataset
    tartanair.py    — TartanAirGaussianDataset
    hypersim.py     — HyperSimGaussianDataset
    srgd.py         — SRGDGaussianDataset
    mixed.py        — MixedGaussianDataset (weighted multi-source sampler)
"""

from .base import (
    CANVAS_CHANNELS,
    DEPTH_CHANNELS,
    GaussianDataset,
    GaussianTrainingExample,
    LR_CHANNELS,
    MOTION_CHANNELS,
    NORMAL_CHANNELS,
    TOTAL_INPUT_CHANNELS,
    collate_examples,
)
from .hypersim import HyperSimGaussianDataset
from .lr_synthesis import (
    EngineAliasedLRSynth,
    area_downsample,
    apply_jitter,
    halton_jitter,
    jpeg_artifact,
    taa_blur_approx,
)
from .mixed import DEFAULT_WEIGHTS, MixedGaussianDataset
from .sintel import SintelGaussianDataset
from .srgd import SRGDGaussianDataset
from .tartanair import TartanAirGaussianDataset

__all__ = [
    # base
    "GaussianTrainingExample",
    "GaussianDataset",
    "collate_examples",
    "LR_CHANNELS",
    "DEPTH_CHANNELS",
    "MOTION_CHANNELS",
    "NORMAL_CHANNELS",
    "CANVAS_CHANNELS",
    "TOTAL_INPUT_CHANNELS",
    # engine-aliased LR synthesis
    "EngineAliasedLRSynth",
    "halton_jitter",
    "apply_jitter",
    "area_downsample",
    "taa_blur_approx",
    "jpeg_artifact",
    # sources
    "SintelGaussianDataset",
    "TartanAirGaussianDataset",
    "HyperSimGaussianDataset",
    "SRGDGaussianDataset",
    # mixer
    "MixedGaussianDataset",
    "DEFAULT_WEIGHTS",
]
