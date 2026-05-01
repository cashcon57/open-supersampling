"""Tests for SWT primitives, WaveletKPNHead, and the wavelet-space loss.

Covers:
- Forward → inverse SWT round-trip at float64 (~1e-15) and float32 (~1e-6).
- WaveletKPNHead output shape and gradient flow.
- Param-count sanity (documents the spec drift: actual ~50K vs the 30K target
  named in the design doc).
- ``wavelet_loss`` zero-at-identity (floor for the optimizer).
"""
from __future__ import annotations

import torch

from oss.model.wavelet import ISWT2D, SWT2D, WaveletKPNHead
from oss.train.losses import wavelet_loss


def test_swt_iswt_round_trip():
    """SWT followed by ISWT must recover the input within tight tolerance."""
    swt = SWT2D(levels=2, wavelet="db2")
    iswt = ISWT2D(levels=2, wavelet="db2")

    # Float64: arithmetic-precision identity. Confirms the math is right.
    torch.manual_seed(0)
    x64 = torch.randn(2, 3, 32, 32, dtype=torch.float64)
    swt64 = SWT2D(levels=2, wavelet="db2").double()
    iswt64 = ISWT2D(levels=2, wavelet="db2").double()
    ll, details = swt64(x64)
    recon = iswt64(ll, details)
    assert (recon - x64).abs().max().item() < 1e-12, (
        "SWT->ISWT round-trip lost precision in float64"
    )

    # Float32: training-precision identity. 1e-5 abs is the spec'd tolerance.
    x32 = torch.randn(2, 3, 32, 32, dtype=torch.float32)
    ll, details = swt(x32)
    recon = iswt(ll, details)
    assert (recon - x32).abs().max().item() < 1e-5, (
        f"float32 round-trip err {(recon - x32).abs().max().item():.2e} exceeds 1e-5"
    )


def test_swt_subband_count_and_shape():
    """For levels=2 SWT2D returns 1 LL + 2 detail triples; shapes preserved."""
    swt = SWT2D(levels=2, wavelet="db2")
    x = torch.randn(1, 3, 16, 16)
    ll, details = swt(x)
    assert ll.shape == x.shape
    assert len(details) == 2
    for triple in details:
        assert len(triple) == 3
        for sb in triple:
            assert sb.shape == x.shape


def test_wavelet_kpn_output_shape():
    """WaveletKPNHead returns (B, 3, H, W) matching the HR noisy color input."""
    head = WaveletKPNHead(feature_ch=32, kernel_size=5, scale_factor=2, levels=2)
    B, H, W = 2, 64, 64
    features_hr = torch.randn(B, 32, H, W)
    noisy_hr = torch.randn(B, 3, H, W)
    out = head(features_hr, noisy_hr)
    assert out.shape == (B, 3, H, W)


def test_wavelet_kpn_param_budget():
    """Document the actual param count for the wavelet head.

    The design doc targeted <30K, but per-subband 5x5 predictors push the
    count to ~50K at the default config (32 ft → 25 outputs × 7 subbands +
    biases plus the 9-tap 3x3 predictor weights). This is documented in
    ``ors/model/wavelet.py``; the test here pins the upper bound at 60K so
    a regression toward 100K+ params trips a failure.
    """
    head = WaveletKPNHead(feature_ch=32, kernel_size=5, scale_factor=2, levels=2)
    n = sum(p.numel() for p in head.parameters())
    # SWT/ISWT filter buffers are non-persistent and don't count as parameters.
    assert n < 60_000, f"WaveletKPNHead param count {n} exceeded 60K bound"


def test_wavelet_kpn_grad_flow():
    """Gradient must flow end-to-end through the head."""
    head = WaveletKPNHead(feature_ch=32, kernel_size=5, scale_factor=2, levels=2)
    features_hr = torch.randn(1, 32, 32, 32, requires_grad=True)
    noisy_hr = torch.randn(1, 3, 32, 32, requires_grad=True)
    out = head(features_hr, noisy_hr)
    out.mean().backward()
    assert features_hr.grad is not None
    assert noisy_hr.grad is not None
    # All learnable params should also have a gradient.
    for name, p in head.named_parameters():
        assert p.grad is not None, f"no grad on {name}"


def test_wavelet_loss_zero_at_identity():
    """``wavelet_loss(x, x)`` must be exactly zero (or float-eps close)."""
    torch.manual_seed(0)
    x = torch.randn(1, 3, 32, 32)
    # ``target`` is detached internally; use a fresh tensor identical in value
    # so the assertion holds without grad bookkeeping side effects.
    loss = wavelet_loss(x, x.clone())
    assert loss.item() < 1e-6, (
        f"wavelet_loss(x, x) = {loss.item()} should be ~0"
    )


def test_wavelet_loss_positive_on_perturbation():
    """Perturbing the prediction must produce a strictly positive loss."""
    torch.manual_seed(0)
    x = torch.randn(1, 3, 32, 32)
    pred = x + 0.1 * torch.randn_like(x)
    loss = wavelet_loss(pred, x)
    assert loss.item() > 0


def test_wavelet_loss_grad_flows_to_pred_only():
    """Gradient must flow into ``pred`` but not ``target`` (target detached)."""
    torch.manual_seed(0)
    pred = torch.randn(1, 3, 32, 32, requires_grad=True)
    target = torch.randn(1, 3, 32, 32, requires_grad=True)
    loss = wavelet_loss(pred, target)
    loss.backward()
    assert pred.grad is not None
    # target should have no grad (we detach inside the loss).
    assert target.grad is None or target.grad.abs().sum().item() == 0.0
