"""OSS-Gaussian network module — Sprint 4 / T4.1, T4.2.

Public API:
- ``CovariancePriorBank``     — fixed/learnable bank of 2D covariance shapes.
- ``CovarianceEntry``         — describing one bank entry.
- ``default_bank_16``         — the 16-entry default vocabulary.
- ``GaussianParamNetwork``    — small CNN predicting tile-wise raw params.
- ``param_net_for_tier``      — factory for Pico/Lite/Standard/Ultra tiers.
- ``TIER_CONFIGS``            — channel/K-per-tile table per hardware tier.
- ``OutputHead``              — decode raw tensor → renderer-ready GaussianBatch.
- ``DecodedParams``           — typed batched output of ``OutputHead.decode``.
"""

from oss.gaussian.network.prior_bank import (
    CovarianceEntry,
    CovariancePriorBank,
    default_bank_16,
)
from oss.gaussian.network.param_net import (
    DEFAULT_BANK_SIZE,
    DEFAULT_K_PER_TILE,
    DEFAULT_TILE_SIZE,
    GaussianParamNetwork,
    TIER_CONFIGS,
    TierConfig,
    param_net_for_tier,
    per_gaussian_channels,
)
from oss.gaussian.network.output_head import DecodedParams, OutputHead

__all__ = [
    "CovariancePriorBank",
    "CovarianceEntry",
    "default_bank_16",
    "GaussianParamNetwork",
    "TierConfig",
    "TIER_CONFIGS",
    "param_net_for_tier",
    "per_gaussian_channels",
    "DEFAULT_BANK_SIZE",
    "DEFAULT_K_PER_TILE",
    "DEFAULT_TILE_SIZE",
    "OutputHead",
    "DecodedParams",
]
