"""End-to-end training-step integration test for v7.

Synthesizes a single trajectory pair (frame N, alpha=0.5 GT, frame
N+1) without touching TartanAir on disk; runs through V7Model
backward / optimizer.step like the real trainer would.

If this test passes, the v7 stack composes for ACTUAL training (modulo
the dataset path which has its own unit tests).
"""
from __future__ import annotations

import torch

from oss.sr.v7.model import V7Config, V7Model
from oss.sr.v7.losses import oss_fx_loss


def _synthesize_9ch_input(B: int, H_lr: int, W_lr: int) -> torch.Tensor:
    """LR (3) + depth (1) + motion (2) + normals (3) = 9 channels."""
    return torch.rand((B, 9, H_lr, W_lr))


def test_full_v7_training_step_runs_without_errors():
    cfg = V7Config(
        in_channels=9, scale=2, feat_dim=16, latent_rank=8,
        canvas_capacity=512, backbone_kind="placeholder",
        backbone_blocks=1, enable_spawner=True,
        spawner_k_per_tile=2, spawner_tile_size=8,
    )
    model = V7Model(cfg).train(True)
    model.allocate_canvas("cpu")

    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Synthesize one triplet
    H_lr, W_lr = 8, 16
    H_hr, W_hr = H_lr * 2, W_lr * 2
    n_lr_in = _synthesize_9ch_input(1, H_lr, W_lr)
    np1_lr_in = _synthesize_9ch_input(1, H_lr, W_lr)
    np1_gt = torch.rand((1, 3, H_hr, W_hr))
    n_half_gt = torch.rand((1, 3, H_hr, W_hr))

    # Forward (matches trainer's two-frame flow)
    model.reset_state("cpu")
    out_n = model(n_lr_in, t_query=0.0, spawn_at_t=0.0)
    out_np1 = model(np1_lr_in, t_query=2.0, spawn_at_t=2.0)
    out_inter = model(np1_lr_in, t_query=1.0)

    # Loss
    loss, parts = oss_fx_loss(
        out_main=out_np1, gt_main=np1_gt,
        out_inter_list=[out_inter], gt_inter_list=[n_half_gt],
        lambda_charbonnier=1.0,
        lambda_lpips=0.0,       # disable to skip lpips dependency
        lambda_fg=1.0,
        lambda_fg_lpips=0.0,
        lambda_temp_consistency=0.0,
    )

    # Backward + step
    optim.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
    optim.step()

    # Sanity checks
    assert torch.isfinite(loss).item()
    assert "sr_charbonnier" in parts
    assert "fg_charbonnier" in parts
    assert parts["total"] > 0.0


def test_full_v7_training_step_with_hat_tiny_backbone():
    """Same flow but with the real HAT-Tiny backbone (transformer)."""
    cfg = V7Config(
        in_channels=9, scale=2, feat_dim=32, latent_rank=8,
        canvas_capacity=512, backbone_kind="hat_tiny",
        enable_spawner=True,
        spawner_k_per_tile=2, spawner_tile_size=16,
    )
    model = V7Model(cfg).train(True)
    model.allocate_canvas("cpu")
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # HAT needs LR dims divisible by window_size (16); use 16x32 LR -> 32x64 HR
    H_lr, W_lr = 16, 32
    H_hr, W_hr = H_lr * 2, W_lr * 2
    n_lr_in = _synthesize_9ch_input(1, H_lr, W_lr)
    np1_lr_in = _synthesize_9ch_input(1, H_lr, W_lr)
    np1_gt = torch.rand((1, 3, H_hr, W_hr))
    n_half_gt = torch.rand((1, 3, H_hr, W_hr))

    model.reset_state("cpu")
    out_n = model(n_lr_in, t_query=0.0, spawn_at_t=0.0)
    out_np1 = model(np1_lr_in, t_query=2.0, spawn_at_t=2.0)
    out_inter = model(np1_lr_in, t_query=1.0)

    loss, _ = oss_fx_loss(
        out_main=out_np1, gt_main=np1_gt,
        out_inter_list=[out_inter], gt_inter_list=[n_half_gt],
        lambda_charbonnier=1.0, lambda_lpips=0.0,
        lambda_fg=1.0, lambda_fg_lpips=0.0,
        lambda_temp_consistency=0.0,
    )
    optim.zero_grad()
    loss.backward()
    optim.step()
    assert torch.isfinite(loss).item()


def test_canvas_grows_through_two_spawns_during_step():
    """Verify the per-step spawn calls actually accumulate canvas
    content over the two-frame forward pass."""
    cfg = V7Config(
        in_channels=9, scale=2, feat_dim=16, latent_rank=8,
        canvas_capacity=2048, backbone_kind="placeholder",
        backbone_blocks=1, enable_spawner=True,
        spawner_k_per_tile=2, spawner_tile_size=8,
    )
    model = V7Model(cfg).train(False)
    model.allocate_canvas("cpu")
    H_lr, W_lr = 8, 16
    n_lr_in = _synthesize_9ch_input(1, H_lr, W_lr)
    np1_lr_in = _synthesize_9ch_input(1, H_lr, W_lr)

    model.reset_state("cpu")
    assert model.canvas.count == 0
    with torch.no_grad():
        _ = model(n_lr_in, t_query=0.0, spawn_at_t=0.0)
    count_after_n = model.canvas.count
    assert count_after_n > 0

    with torch.no_grad():
        _ = model(np1_lr_in, t_query=2.0, spawn_at_t=2.0)
    count_after_np1 = model.canvas.count
    assert count_after_np1 == count_after_n * 2, (
        f"expected canvas to double after second spawn; got "
        f"{count_after_np1} from {count_after_n}"
    )
