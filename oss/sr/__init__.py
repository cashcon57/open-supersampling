"""OSS-SR — CNN super-resolver module.

Forked from the Gaussian splat track on 2026-05-02 after Sprint 4 confirmed
that 2D Gaussian splats cannot compete with a small CNN for single-image SR.

Public API
----------
    SRCNNSimple    V0 backbone: small residual CNN, tier-scaled.
    SRRRDB         V1 backbone: ESRGAN-style RRDB, higher quality.
    build_sr_model Factory: returns the requested model given kind + tier.

Input contract (both models)
-----------------------------
    (B, 12, h, w)  — 12 channels: LR-RGB(3) + depth(1) + motion(2)
                                   + normals(3) + canvas_hint(3).
    First 3 channels must be LR RGB for the internal bicubic skip.

Output: (B, 3, 2h, 2w) unclamped HR RGB.  Caller applies .clamp(0, 1).
"""

from __future__ import annotations

from oss.sr.cnn import SR_TIER_CONFIGS, SRCNNSimple, srcnn_for_tier
from oss.sr.rrdb import SRRRDB


def build_sr_model(
    model_kind: str,
    tier: str,
    in_channels: int = 12,
    scale: int = 2,
) -> "SRCNNSimple | SRRRDB":
    """Factory: build an SR model by kind and hardware tier.

    Args:
        model_kind:   ``"simple"`` → SRCNNSimple,  ``"rrdb"`` → SRRRDB.
        tier:         Hardware tier — ``"pico"``, ``"lite"``, or ``"standard"``.
                      For RRDB the tier selects hidden-channel count from the
                      same table as SRCNNSimple.
        in_channels:  Input channel count (default 12).
        scale:        SR scale factor (default 2).

    Returns:
        Configured, un-trained SR model.

    Raises:
        ValueError: On unknown model_kind or tier.
    """
    if tier not in SR_TIER_CONFIGS:
        raise ValueError(
            f"Unknown SR tier {tier!r}. Valid choices: {sorted(SR_TIER_CONFIGS)}"
        )

    hidden, n_blocks = SR_TIER_CONFIGS[tier]

    if model_kind == "simple":
        return SRCNNSimple(
            in_channels=in_channels,
            scale=scale,
            hidden=hidden,
            n_blocks=n_blocks,
        )
    elif model_kind == "rrdb":
        # For RRDB, n_blocks from the tier table becomes n_rrdb.
        # Growth is fixed at hidden//2 (at least 8) to keep params proportional.
        growth = max(8, hidden // 2)
        return SRRRDB(
            in_channels=in_channels,
            scale=scale,
            hidden=hidden,
            n_rrdb=n_blocks,
            growth=growth,
        )
    else:
        raise ValueError(
            f"Unknown SR model kind {model_kind!r}. Valid choices: 'simple', 'rrdb'."
        )


__all__ = ["SRCNNSimple", "SRRRDB", "build_sr_model"]
