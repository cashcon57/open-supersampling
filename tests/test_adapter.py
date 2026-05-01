import pytest
import torch

from oss.handoff.contract import (
    HandoffContract,
    HandoffContractError,
    validate_handoff,
)
from oss.model import ORD, ORU
from oss.model.adapter import PairedORS


def test_handoff_contract_constants():
    assert HandoffContract.VERSION == 1
    assert HandoffContract.CHANNELS == 32
    assert HandoffContract.DTYPE == torch.float16


def test_validate_handoff_accepts_valid():
    feats = torch.zeros(1, 32, 16, 16, dtype=torch.float16)
    validate_handoff(feats)  # no raise


def test_validate_handoff_rejects_wrong_dtype():
    feats = torch.zeros(1, 32, 16, 16, dtype=torch.float32)
    with pytest.raises(HandoffContractError, match="dtype"):
        validate_handoff(feats)


def test_validate_handoff_rejects_wrong_channels():
    feats = torch.zeros(1, 16, 16, 16, dtype=torch.float16)
    with pytest.raises(HandoffContractError, match="channels"):
        validate_handoff(feats)


def test_validate_handoff_rejects_wrong_ndim():
    feats = torch.zeros(32, 16, 16, dtype=torch.float16)
    with pytest.raises(HandoffContractError, match="4D"):
        validate_handoff(feats)


def test_paired_rejects_non_features_oru():
    ord_model = ORD(tier="standard")
    oru_rgb = ORU(input_mode="rgb", scale_factor=2.0, tier="standard")
    with pytest.raises(ValueError, match="features"):
        PairedORS(ord_model, oru_rgb)


def test_paired_end_to_end():
    ord_model = ORD(tier="standard").train(False)
    oru_model = ORU(input_mode="features", scale_factor=2.0, tier="standard").train(False)
    pair = PairedORS(ord_model, oru_model)

    B, H, W = 1, 32, 32
    rgb_lo, rgb_hi = pair(
        noisy=torch.randn(B, 3, H, W),
        aux=torch.randn(B, 11, H, W),
        history=torch.randn(B, 3, H, W),
        depth=torch.randn(B, 1, H, W),
        motion=torch.randn(B, 2, H, W),
    )
    assert rgb_lo.shape == (B, 3, H, W)
    assert rgb_hi.shape == (B, 3, H * 2, W * 2)
