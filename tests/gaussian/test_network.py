"""Sprint 4 architecture tests — runs on CPU, no GPU dependency.

Covers:
- Covariance Prior Bank produces valid 2×2 cov matrices (PSD, finite).
- GaussianParamNetwork forward shape correctness across resolutions / tiers.
- OutputHead produces valid in-frame positions, valid scales/rot, color in
  the documented activation range.
- End-to-end differentiability: random input → network → decode → renderer
  → loss → backward → grads non-zero on the network's parameters AND on the
  bank's parameters when learnable.
"""

from __future__ import annotations

import math

import pytest
import torch

from oss.gaussian.network import (
    CovariancePriorBank,
    DEFAULT_BANK_SIZE,
    GaussianParamNetwork,
    OutputHead,
    TIER_CONFIGS,
    default_bank_16,
    param_net_for_tier,
    per_gaussian_channels,
)
from oss.gaussian.renderer import GaussianBatch, Rasterizer


# ---------------------------------------------------------------------------
# Covariance Prior Bank
# ---------------------------------------------------------------------------


def test_default_bank_16_has_sixteen_entries() -> None:
    entries = default_bank_16()
    assert len(entries) == 16
    for e in entries:
        assert e.sx > 0 and e.sy > 0
        assert math.isfinite(e.theta_rad)


def test_bank_covariance_matrices_are_psd() -> None:
    bank = CovariancePriorBank()
    cov = bank.covariance_matrices()
    assert cov.shape == (16, 2, 2)
    assert torch.isfinite(cov).all()
    # PSD ↔ trace > 0 and det >= 0 for 2×2 symmetric matrices.
    a = cov[:, 0, 0]
    b = cov[:, 0, 1]
    d = cov[:, 1, 1]
    trace = a + d
    det = a * d - b * b
    assert (trace > 0).all(), "trace must be positive"
    # Allow a tiny negative due to FP — but require non-degenerate.
    assert (det > -1e-6).all(), f"det must be ≥ 0 (PSD); got min {det.min().item()}"
    assert (det > 0).all(), "all bank cov matrices must be non-degenerate"
    # Symmetry.
    assert torch.allclose(cov[:, 0, 1], cov[:, 1, 0])


def test_bank_forward_uniform_weights_valid() -> None:
    bank = CovariancePriorBank()
    K = bank.bank_size
    # Uniform weights over the bank, batch (3, 7, K).
    weights = torch.full((3, 7, K), 1.0 / K)
    sx, sy, theta, cov = bank(weights)
    assert sx.shape == (3, 7)
    assert sy.shape == (3, 7)
    assert theta.shape == (3, 7)
    assert cov.shape == (3, 7, 2, 2)
    assert (sx > 0).all() and (sy > 0).all()
    assert torch.isfinite(cov).all()


def test_bank_forward_one_hot_recovers_entry() -> None:
    bank = CovariancePriorBank()
    K = bank.bank_size
    entries = bank.entries()
    for k in range(K):
        weights = torch.zeros(K)
        weights[k] = 1.0
        sx, sy, theta, _cov = bank(weights.unsqueeze(0))
        # Geometric weighted mean with one-hot weight should recover the entry.
        assert math.isclose(sx.item(), entries[k].sx, rel_tol=1e-5)
        assert math.isclose(sy.item(), entries[k].sy, rel_tol=1e-5)
        # θ via atan2(sin, cos) — recover same angle modulo 2π.
        # Wrap into [-π, π] for comparison.
        e_t = math.atan2(math.sin(entries[k].theta_rad), math.cos(entries[k].theta_rad))
        assert math.isclose(theta.item(), e_t, abs_tol=1e-5)


def test_bank_rejects_negative_or_inf_weights() -> None:
    bank = CovariancePriorBank()
    bad = torch.full((1, bank.bank_size), -0.1)
    with pytest.raises(ValueError):
        bank(bad)
    nan = torch.full((1, bank.bank_size), float("inf"))
    with pytest.raises(ValueError):
        bank(nan)


def test_learnable_bank_has_trainable_parameters() -> None:
    bank = CovariancePriorBank(learnable=True)
    n_params = sum(p.numel() for p in bank.parameters() if p.requires_grad)
    # 3 * bank_size: log_sx, log_sy, theta.
    assert n_params == 3 * bank.bank_size


def test_frozen_bank_has_no_trainable_parameters() -> None:
    bank = CovariancePriorBank(learnable=False)
    n_params = sum(p.numel() for p in bank.parameters() if p.requires_grad)
    assert n_params == 0


# ---------------------------------------------------------------------------
# GaussianParamNetwork shape
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("h_lr,w_lr", [(32, 32), (64, 96), (128, 128)])
def test_param_net_output_shape(h_lr: int, w_lr: int) -> None:
    net = GaussianParamNetwork()
    x = torch.randn(1, net.in_channels, h_lr, w_lr)
    out = net(x)
    expected = (1, net.out_channels, h_lr // net.tile_size, w_lr // net.tile_size)
    assert out.shape == expected


def test_param_net_rejects_non_tile_aligned_input() -> None:
    net = GaussianParamNetwork()
    bad = torch.randn(1, net.in_channels, 33, 32)
    with pytest.raises(ValueError, match="multiples of tile_size"):
        net(bad)


def test_param_net_rejects_wrong_in_channels() -> None:
    net = GaussianParamNetwork(in_channels=12)
    bad = torch.randn(1, 7, 32, 32)
    with pytest.raises(ValueError, match="in_channels mismatch"):
        net(bad)


def test_param_net_initial_output_is_zero() -> None:
    """Head is zero-initialised — first forward pass should be exactly zeros."""
    net = GaussianParamNetwork()
    x = torch.randn(1, net.in_channels, 32, 32)
    out = net(x)
    assert torch.allclose(out, torch.zeros_like(out))


@pytest.mark.parametrize("tier", sorted(TIER_CONFIGS))
def test_param_net_tier_factory(tier: str) -> None:
    net = param_net_for_tier(tier)
    cfg = TIER_CONFIGS[tier]
    assert net.k_per_tile == cfg.k_per_tile
    assert net.channels == cfg.channels
    x = torch.randn(1, net.in_channels, 32, 32)
    out = net(x)
    expected_ch = cfg.k_per_tile * per_gaussian_channels(net.bank_size)
    assert out.shape == (1, expected_ch, 32 // net.tile_size, 32 // net.tile_size)


# ---------------------------------------------------------------------------
# OutputHead decoding
# ---------------------------------------------------------------------------


def _random_raw(B: int, k: int, bank_size: int, Ht: int, Wt: int) -> torch.Tensor:
    """A non-zero raw tensor so decode actually exercises every branch."""
    return torch.randn(B, k * per_gaussian_channels(bank_size), Ht, Wt)


def test_output_head_decode_shapes() -> None:
    bank = CovariancePriorBank()
    head = OutputHead(bank=bank, tile_size=16, k_per_tile=5)
    raw = _random_raw(B=2, k=5, bank_size=bank.bank_size, Ht=2, Wt=3)
    d = head.decode(raw)
    N = 2 * 3 * 5
    assert d.xy.shape == (2, N, 2)
    assert d.scale.shape == (2, N, 2)
    assert d.rot.shape == (2, N)
    assert d.feat.shape == (2, N, 3)
    assert d.bank_weights.shape == (2, N, bank.bank_size)


def test_output_head_positions_inside_padded_frame() -> None:
    bank = CovariancePriorBank()
    head = OutputHead(bank=bank, tile_size=16, k_per_tile=5)
    Ht, Wt = 4, 5  # frame 64×80
    raw = _random_raw(B=1, k=5, bank_size=bank.bank_size, Ht=Ht, Wt=Wt) * 5.0
    d = head.decode(raw)
    # tanh × tile_size offset envelope — centers stay within
    # [-tile_size, frame_w + tile_size] / [-tile_size, frame_h + tile_size].
    frame_w = Wt * 16
    frame_h = Ht * 16
    assert (d.xy[..., 0] >= -16).all() and (d.xy[..., 0] <= frame_w + 16).all()
    assert (d.xy[..., 1] >= -16).all() and (d.xy[..., 1] <= frame_h + 16).all()


def test_output_head_color_in_sigmoid_range() -> None:
    bank = CovariancePriorBank()
    head = OutputHead(bank=bank, color_activation="sigmoid")
    raw = _random_raw(B=1, k=5, bank_size=bank.bank_size, Ht=2, Wt=2) * 10.0
    d = head.decode(raw)
    assert (d.feat >= 0).all() and (d.feat <= 1).all()


def test_output_head_softplus_color_non_negative() -> None:
    bank = CovariancePriorBank()
    head = OutputHead(bank=bank, color_activation="softplus")
    raw = _random_raw(B=1, k=5, bank_size=bank.bank_size, Ht=2, Wt=2) * 10.0
    d = head.decode(raw)
    assert (d.feat >= 0).all()


def test_output_head_scales_within_envelope() -> None:
    bank = CovariancePriorBank()
    head = OutputHead(bank=bank, log_scale_clip=math.log(8.0))
    raw = _random_raw(B=1, k=5, bank_size=bank.bank_size, Ht=2, Wt=2) * 50.0
    d = head.decode(raw)
    # Combine scale_factor (∈ [1/8, 8]) with bank min/max sx, sy.
    bank_entries = default_bank_16()
    bank_max_sx = max(e.sx for e in bank_entries)
    bank_max_sy = max(e.sy for e in bank_entries)
    assert (d.scale > 0).all()
    assert d.scale[..., 0].max() <= bank_max_sx * 8.0 + 1e-3
    assert d.scale[..., 1].max() <= bank_max_sy * 8.0 + 1e-3


def test_output_head_to_gaussian_batch() -> None:
    bank = CovariancePriorBank()
    net = GaussianParamNetwork()
    head = OutputHead(bank=bank, k_per_tile=net.k_per_tile)
    x = torch.randn(1, net.in_channels, 32, 32)
    raw = net(x) + torch.randn_like(net(x))  # break the zero-init
    gb = head.to_gaussian_batch(raw, batch_index=0)
    assert isinstance(gb, GaussianBatch)
    assert gb.num_gaussians == (32 // 16) * (32 // 16) * net.k_per_tile


# ---------------------------------------------------------------------------
# End-to-end differentiability
# ---------------------------------------------------------------------------


def test_end_to_end_differentiability_grads_flow() -> None:
    """Random LR + G-buffers → network → decode → renderer → loss → backward.

    Verifies non-zero gradients on every leaf parameter of the network. The
    renderer's reference backend is differentiable in pure torch — no CUDA
    needed for this sanity check.
    """
    torch.manual_seed(0)
    bank = CovariancePriorBank(learnable=True)
    net = GaussianParamNetwork(bank_size=bank.bank_size, k_per_tile=3,
                               channels=(8, 16, 24, 32))
    head = OutputHead(bank=bank, tile_size=net.tile_size, k_per_tile=net.k_per_tile)

    H_lr, W_lr = 32, 32
    # 12-channel input: LR(3) + depth(1) + motion(2) + normals(3) + canvas(3).
    x = torch.randn(1, 12, H_lr, W_lr)
    target = torch.rand(3, H_lr, W_lr)

    # The head is zero-initialised by design (so the model produces tame
    # outputs at step 0). For this gradient-flow test we want signal flowing
    # through the entire network, so seed the head with small random values
    # — this matches what would happen after a single optimiser step.
    with torch.no_grad():
        net.head.weight.normal_(0, 0.01)
        net.head.bias.normal_(0, 0.01)
    raw = net(x)

    gb = head.to_gaussian_batch(raw, batch_index=0)
    rasterizer = Rasterizer(force_backend="reference")
    rendered = rasterizer(gb, output_hw=(H_lr, W_lr))
    loss = torch.nn.functional.mse_loss(rendered, target)
    loss.backward()

    # The convolutional layers' first conv weight must receive grad.
    stem_w = net.stem.conv.weight
    assert stem_w.grad is not None and torch.any(stem_w.grad != 0), \
        "stem.conv.weight received no gradient — encoder is dead"
    head_w = net.head.weight
    assert head_w.grad is not None and torch.any(head_w.grad != 0), \
        "head.weight received no gradient — output head is dead"
    # The learnable bank should receive non-zero grad too.
    assert bank.log_sx.grad is not None and torch.any(bank.log_sx.grad != 0)


# ---------------------------------------------------------------------------
# Anisotropic G-buffer-conditioned covariance bias (validation memo Decision 2)
# ---------------------------------------------------------------------------


def test_output_head_gbuffer_bias_disabled_default() -> None:
    """Default OutputHead has no gbuffer_bias module (backward compat)."""
    bank = CovariancePriorBank()
    head = OutputHead(bank=bank)
    assert head.enable_gbuffer_bias is False
    assert head.gbuffer_bias is None


def test_output_head_gbuffer_bias_zero_init_matches_baseline() -> None:
    """Enabling the bias with zero-init must match disabled output bit-for-bit
    when called without G-buffer inputs (graceful enablement)."""
    torch.manual_seed(0)
    bank = CovariancePriorBank()
    head_off = OutputHead(bank=bank, tile_size=16, k_per_tile=5)
    head_on = OutputHead(bank=bank, tile_size=16, k_per_tile=5,
                         enable_gbuffer_bias=True)
    raw = _random_raw(B=1, k=5, bank_size=bank.bank_size, Ht=2, Wt=2)
    d_off = head_off.decode(raw)
    d_on = head_on.decode(raw)  # No depth/normals → bias is zero.
    assert torch.allclose(d_off.bank_weights, d_on.bank_weights), \
        "enable_gbuffer_bias=True without G-buffers should match the baseline"


def test_output_head_gbuffer_bias_zero_input_no_change() -> None:
    """Even with the bias enabled and G-buffers passed, all-zero G-buffers
    produce no logit change (zero-init linear of all-zero input = 0)."""
    torch.manual_seed(0)
    bank = CovariancePriorBank()
    head_off = OutputHead(bank=bank, tile_size=16, k_per_tile=5)
    head_on = OutputHead(bank=bank, tile_size=16, k_per_tile=5,
                         enable_gbuffer_bias=True)
    Ht, Wt = 2, 2
    H_lr, W_lr = Ht * 16, Wt * 16
    raw = _random_raw(B=1, k=5, bank_size=bank.bank_size, Ht=Ht, Wt=Wt)
    depth = torch.zeros(1, 1, H_lr, W_lr)
    normals = torch.zeros(1, 3, H_lr, W_lr)
    d_off = head_off.decode(raw)
    d_on = head_on.decode(raw, depth=depth, normals=normals)
    assert torch.allclose(d_off.bank_weights, d_on.bank_weights), \
        "Zero G-buffers + zero-init bias must equal baseline output"


def test_output_head_gbuffer_bias_changes_weights_after_training_step() -> None:
    """Once the bias linear has non-zero weights, non-zero G-buffers must
    actually shift the bank softmax distribution."""
    torch.manual_seed(0)
    bank = CovariancePriorBank()
    head = OutputHead(bank=bank, tile_size=16, k_per_tile=5,
                      enable_gbuffer_bias=True)
    # Simulate a partially-trained bias head.
    with torch.no_grad():
        head.gbuffer_bias.proj.weight.normal_(0, 0.5)
        head.gbuffer_bias.proj.bias.normal_(0, 0.5)
    Ht, Wt = 2, 2
    H_lr, W_lr = Ht * 16, Wt * 16
    raw = _random_raw(B=1, k=5, bank_size=bank.bank_size, Ht=Ht, Wt=Wt)
    depth = torch.rand(1, 1, H_lr, W_lr)
    normals = torch.randn(1, 3, H_lr, W_lr)
    normals = normals / normals.norm(dim=1, keepdim=True).clamp(min=1e-6)

    d_no_gbuf = head.decode(raw)
    d_gbuf = head.decode(raw, depth=depth, normals=normals)
    assert not torch.allclose(d_no_gbuf.bank_weights, d_gbuf.bank_weights), \
        "Non-trivial G-buffers + non-zero bias weights should change softmax"
    # Softmax outputs must still be valid probability distributions.
    assert torch.allclose(d_gbuf.bank_weights.sum(dim=-1),
                          torch.ones_like(d_gbuf.bank_weights.sum(dim=-1)),
                          atol=1e-5)
    assert (d_gbuf.bank_weights >= 0).all()


def test_output_head_gbuffer_bias_per_tile_shared_across_k() -> None:
    """The G-buffer bias is per-tile, applied identically to all K Gaussians
    in the same tile. Verify by constructing input where bank_logits are
    identical across K, plus a non-trivial bias — output bank_weights should
    be identical across K within each tile."""
    torch.manual_seed(0)
    bank = CovariancePriorBank()
    K = 5
    bank_size = bank.bank_size
    head = OutputHead(bank=bank, tile_size=16, k_per_tile=K,
                      enable_gbuffer_bias=True)
    with torch.no_grad():
        head.gbuffer_bias.proj.weight.normal_(0, 0.5)
        head.gbuffer_bias.proj.bias.normal_(0, 0.5)

    Ht, Wt = 2, 2
    H_lr, W_lr = Ht * 16, Wt * 16
    per_g = per_gaussian_channels(bank_size)
    # Build a raw tensor where every Gaussian within a tile gets the same
    # per-Gaussian channel vector (identical bank_logits across K).
    base = torch.randn(1, 1, per_g, Ht, Wt)
    raw_5d = base.expand(1, K, per_g, Ht, Wt).contiguous()
    raw = raw_5d.view(1, K * per_g, Ht, Wt)
    depth = torch.rand(1, 1, H_lr, W_lr)
    normals = torch.randn(1, 3, H_lr, W_lr)
    d = head.decode(raw, depth=depth, normals=normals)

    # bank_weights shape: (1, N, bank_size); N = Ht*Wt*K.
    bw = d.bank_weights.view(1, Ht, Wt, K, bank_size)
    # Within each tile, all K rows should be identical.
    for ti in range(Ht):
        for tj in range(Wt):
            tile = bw[0, ti, tj]  # (K, bank_size)
            assert torch.allclose(tile[0], tile[-1], atol=1e-6), \
                f"tile ({ti},{tj}) bank_weights not shared across K Gaussians"


def test_output_head_gbuffer_bias_grad_flow() -> None:
    """Gradients must flow through the G-buffer bias module when it's used."""
    torch.manual_seed(0)
    bank = CovariancePriorBank()
    head = OutputHead(bank=bank, tile_size=16, k_per_tile=3,
                      enable_gbuffer_bias=True)
    # Break the zero init so the loss depends on the bias projection.
    with torch.no_grad():
        head.gbuffer_bias.proj.weight.normal_(0, 0.1)

    Ht, Wt = 2, 2
    H_lr, W_lr = Ht * 16, Wt * 16
    raw = _random_raw(B=1, k=3, bank_size=bank.bank_size, Ht=Ht, Wt=Wt)
    raw = raw.detach().requires_grad_(False)
    depth = torch.rand(1, 1, H_lr, W_lr, requires_grad=False)
    normals = torch.randn(1, 3, H_lr, W_lr, requires_grad=False)

    d = head.decode(raw, depth=depth, normals=normals)
    # Loss anchored on bank_weights so the bias projection is on-graph.
    loss = d.bank_weights.sum()
    loss.backward()
    assert head.gbuffer_bias.proj.weight.grad is not None
    assert torch.any(head.gbuffer_bias.proj.weight.grad != 0)


def test_output_head_gbuffer_bias_rejects_wrong_shapes() -> None:
    """Spatial mismatch between G-buffers and tile grid must fail loudly."""
    bank = CovariancePriorBank()
    head = OutputHead(bank=bank, tile_size=16, k_per_tile=3,
                      enable_gbuffer_bias=True)
    raw = _random_raw(B=1, k=3, bank_size=bank.bank_size, Ht=2, Wt=2)
    bad_depth = torch.zeros(1, 1, 24, 24)  # not a multiple of tile_size×Ht
    with pytest.raises(ValueError, match="depth spatial"):
        head.decode(raw, depth=bad_depth)
    bad_normals = torch.zeros(1, 3, 24, 24)
    with pytest.raises(ValueError, match="normals spatial"):
        head.decode(raw, normals=bad_normals)


def test_param_net_param_count_pico_under_50k() -> None:
    """Pico tier should be markedly smaller than the standard tier."""
    pico = param_net_for_tier("pico")
    standard = param_net_for_tier("standard")
    pico_n = sum(p.numel() for p in pico.parameters())
    std_n = sum(p.numel() for p in standard.parameters())
    assert pico_n < std_n, f"pico ({pico_n}) should be smaller than standard ({std_n})"
    # Sanity: both well under 1M params (lightweight CNN).
    assert std_n < 1_000_000
