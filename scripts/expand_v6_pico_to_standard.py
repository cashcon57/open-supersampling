#!/usr/bin/env python3
"""Net2Net expansion: v6-Pico checkpoint -> v6-Standard initial weights.

Function-preserving expansion that grows a trained Pico model into a
Standard-shaped model whose initial output **exactly equals** Pico's
output on the same input. Then continue training Standard from this
warm start to leverage the new capacity.

This is the v1-product cascade: Pico-from-scratch -> Standard via
Net2Net. (v2 product replaces both with Heavy-from-scratch + memo
distillation cascade.)

Reference papers:
  Chen, Goodfellow, Shlens 2015. "Net2Net: Accelerating Learning via
    Knowledge Transfer." arXiv:1511.05641
  Karras et al. 2017. "Progressive Growing of GANs for Improved
    Quality, Stability, and Variation." arXiv:1710.10196

Architecture diff Pico (hat-tiny) -> Standard (hat-small):
  embed_dim    60 -> 120     (Net2WiderNet width-double)
  depth         2 ->   4     (Net2DeeperNet identity-init new HABs)
  num_heads     4 ->   6     (head-split-and-rescale; head_dim 15 -> 20)
  window_size  16 -> 16      (unchanged — no expansion needed)
  mlp_ratio  2.0 -> 2.0      (unchanged)

V6 wrapper components:
  token_dim    32 -> 64      (Net2WiderNet on canvas_to_token + spawner head)
  canvas_capacity   1500 -> 5000 (just larger int, non-parametric)
  cross_attention_heads 4 -> 6 (head-rescale, see HAT note above)
  composite_head — middle layer width scales with feat_dim (auto-handled)

USAGE:
  ./scripts/expand_v6_pico_to_standard.py \\
      --pico-ckpt  <train-host-data>/checkpoints/srcnn-v6-pico-001/step-00250000.pt \\
      --output     <train-host-data>/checkpoints/srcnn-v6-standard-001/step-00000000-from-pico.pt

After this writes the seed checkpoint, launch normal v6 training with
``--output-dir <train-host-data>/checkpoints/srcnn-v6-standard-001`` and the trainer's
auto-resume will pick it up at step 0 with Pico-quality output. Continue
training to leverage the doubled capacity.

VERIFICATION (assert at end of script): expanded(lr) ~= pico(lr)
  ε ≤ 1e-3 mean abs diff on a fixed seed batch — function-preserving
  property of Net2Net.

IMPLEMENTATION NOTES (HAT-specific gotchas):
  1. WindowAttention's qkv linear has shape (3*dim, dim). Doubling dim
     means the qkv weight needs 4x more entries; use Net2WiderNet
     duplicate-and-rescale on both axes.
  2. Relative position bias tables in HAT depend on (window_size**2)
     and num_heads. When num_heads changes, the bias table must be
     interpolated (or replicated for new heads with rescaling). Tricky.
  3. LayerNorm has learnable scale + bias of dim D. When D doubles,
     duplicate the entries — the scale stays unit because we also
     halved the contributing weights elsewhere.
  4. New HAB blocks (depth 2 -> 4) initialize their attention QKV +
     MLP weights to ZERO so the block's residual contribution is zero
     at expansion. Output is unchanged from Pico's depth-2 forward.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn

# Allow ``python scripts/...`` from a system Python without installing.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from oss.sr.v6.model import V6Config, V6Model  # noqa: E402

log = logging.getLogger("expand_v6_pico_to_standard")


# ---------------------------------------------------------------------------
# Net2Net atomic operations
# ---------------------------------------------------------------------------


def net2wider_linear(
    weight: torch.Tensor,    # (out_old, in_old)
    bias: torch.Tensor | None,
    out_new: int | None = None,
    in_new: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Net2WiderNet on a Linear layer.

    To preserve the function, when widening the OUTPUT dim we duplicate
    each output channel (with no rescaling at this layer); the consuming
    layer's matching input slots get ``1/k`` rescaling so the duplicate
    activations sum back to the original.

    When widening the INPUT dim we average-split each input channel
    across new slots (no consuming-layer rescaling needed).
    """
    out_old, in_old = weight.shape
    out_new = out_new or out_old
    in_new = in_new or in_old
    if out_new < out_old or in_new < in_old:
        raise ValueError("net2wider only supports growing dimensions")

    new_w = weight.new_zeros((out_new, in_new))
    new_w[:out_old, :in_old] = weight

    # Duplicate output channels to fill the new rows (random pick from
    # existing rows); upstream consumer must rescale by 1/replication_count.
    if out_new > out_old:
        gen = torch.Generator(device=weight.device).manual_seed(0)
        idx = torch.randint(0, out_old, (out_new - out_old,), generator=gen)
        new_w[out_old:] = weight[idx]

    # Average-split input channels — when a downstream layer's input has
    # been widened by net2wider on the previous layer's output, this
    # layer's input weights get the matching index pattern with rescale.
    # Caller is responsible for telling us which input slots to rescale.

    new_b = None
    if bias is not None:
        new_b = bias.new_zeros((out_new,))
        new_b[:out_old] = bias
        if out_new > out_old:
            gen = torch.Generator(device=bias.device).manual_seed(0)
            idx = torch.randint(0, out_old, (out_new - out_old,), generator=gen)
            new_b[out_old:] = bias[idx]
    return new_w, new_b


def net2wider_conv2d(
    weight: torch.Tensor,    # (out_C, in_C, kH, kW)
    bias: torch.Tensor | None,
    out_new: int | None = None,
    in_new: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Net2WiderNet on a Conv2d. Same logic as Linear, but on (out, in)
    of the (out, in, kH, kW) tensor."""
    out_old, in_old, kh, kw = weight.shape
    out_new = out_new or out_old
    in_new = in_new or in_old
    new_w = weight.new_zeros((out_new, in_new, kh, kw))
    new_w[:out_old, :in_old] = weight
    if out_new > out_old:
        gen = torch.Generator(device=weight.device).manual_seed(0)
        idx = torch.randint(0, out_old, (out_new - out_old,), generator=gen)
        new_w[out_old:, :in_old] = weight[idx]
    new_b = None
    if bias is not None:
        new_b = bias.new_zeros((out_new,))
        new_b[:out_old] = bias
        if out_new > out_old:
            gen = torch.Generator(device=bias.device).manual_seed(0)
            idx = torch.randint(0, out_old, (out_new - out_old,), generator=gen)
            new_b[out_old:] = bias[idx]
    return new_w, new_b


def net2deeper_block(
    block: nn.Module,
) -> None:
    """Net2DeeperNet — initialize a fresh block's parameters to identity.

    For HAT's HAB block (attn + mlp with residual connections), this
    means: zero the output projection weights of attn and mlp, leaving
    biases at their kaiming-default. The residual ``x = x + attn(x)``
    becomes ``x = x + 0 = x``, identity. Same for mlp.

    Caller passes a block whose state_dict isn't yet loaded from Pico —
    after this function it's a "passes through" identity layer ready to
    be trained.
    """
    # Zero the last projection of attention (so attn output = 0)
    for name, module in block.named_modules():
        if name.endswith("proj") or name.endswith("fc2"):
            if hasattr(module, "weight"):
                nn.init.zeros_(module.weight)
            if hasattr(module, "bias") and module.bias is not None:
                nn.init.zeros_(module.bias)


# ---------------------------------------------------------------------------
# HAT-specific expansion (the hard part)
# ---------------------------------------------------------------------------


def expand_hat_tiny_to_small(
    pico_hat_state: dict[str, torch.Tensor],
    standard_hat: nn.Module,
) -> None:
    """Copy + width-expand HAT-Tiny weights into HAT-Small structure.

    Per-layer expansions:
      patch_embed   Conv2d(in_ch, 60, 3) -> Conv2d(in_ch, 120, 3)
        -> net2wider_conv2d on out dim only.

      blocks 0..1 (Pico) -> blocks 0..1 (Standard, expanded width):
        - WindowAttention.qkv: Linear(60, 180) -> Linear(120, 360)
          (qkv outputs 3*dim, head reshape happens after)
          -> net2wider_linear with both in and out widened. With heads
          changing 4 -> 6, also need to handle head_dim 15 -> 20: this
          requires re-blocking the qkv tensor by attention head, then
          duplicating each head's slice with rescaling.
        - WindowAttention.proj: Linear(60, 60) -> Linear(120, 120)
        - mlp.fc1, mlp.fc2: matched widening
        - relative_position_bias_table: shape (W*2-1)^2 x num_heads ->
          interpolate along the heads axis (not trivial — see notes
          below), and KEEP window_size constant so spatial positions
          unchanged.

      blocks 2..3 (Standard only, no Pico counterpart):
        - net2deeper_block — initialize to identity.

      out_proj  Conv2d(60, 60, 3) -> Conv2d(120, 120, 3)
        -> net2wider_conv2d on both in and out.

    KNOWN HARD CASE — relative_position_bias_table head expansion:
      The bias is a learned (window_h * window_w * num_heads) tensor.
      Going 4 heads -> 6 heads means inserting 2 new heads' worth of
      bias rows. Net2Net doesn't have a clean recipe for this; current
      practice:
        a) duplicate two existing heads' bias rows (with halved scale
           on the originals to keep sum the same), OR
        b) zero-init the new head rows and accept a small function-
           preservation error (~0.1-0.5 dB at expansion).
      This script uses (a) — exact identity preserved on heads 0..3 and
      heads 4..5 are duplicates of heads 0..1 with halved magnitude.

    NOTE: this is NOT just a per-tensor expansion. The QKV's head-dim
    change is the trickiest piece. See `expand_qkv_with_head_change` below.
    """
    raise NotImplementedError(
        "Per-layer HAT expansion code is sketched in the docstring above. "
        "Implementation requires:"
        "\n  1. Iterate Pico HAT state_dict by layer name."
        "\n  2. For each tensor, dispatch to net2wider_conv2d / net2wider_linear "
        "with target dims read from standard_hat's named_parameters."
        "\n  3. Special-case the WindowAttention.qkv tensor for head-count change."
        "\n  4. Special-case relative_position_bias_table for head expansion."
        "\n  5. Identity-init blocks 2 and 3 via net2deeper_block."
        "\n  6. Load expanded state_dict into standard_hat with strict=True."
        "\nEstimated implementation: ~200 LOC, ~3-4 days research-quality work."
    )


def expand_qkv_with_head_change(
    qkv_weight: torch.Tensor,    # (3*dim_old, dim_old) = (180, 60) for tiny
    qkv_bias: torch.Tensor | None,
    dim_old: int,
    num_heads_old: int,
    dim_new: int,
    num_heads_new: int,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Expand a QKV linear when both total dim AND head count change.

    Logic:
      1. Reshape qkv_weight from (3*dim_old, dim_old) to
         (3, num_heads_old, head_dim_old, dim_old) to expose the head
         structure.
      2. Net2Wider on the head_dim_old -> head_dim_new dim.
      3. Net2Wider on the dim_old (input) -> dim_new dim.
      4. Add (num_heads_new - num_heads_old) new heads by duplicating
         existing head slices with halved scale on the originals so
         output magnitudes match.
      5. Reshape back to (3 * dim_new, dim_new) — but careful: the
         reshape index ordering must match nn.Linear's qkv access pattern
         (see hat.py line 75-82 where qkv is reshaped + permuted).
    """
    raise NotImplementedError(
        "QKV head-change expansion is the trickiest part of Pico->Standard. "
        "Reference: SwinTransformer-style window-attention has the same head-dim "
        "coupling. Standard practice in Net2Net papers for transformer scaling: "
        "1) reshape qkv as (3, heads, head_dim, in_dim); 2) net2wider per axis; "
        "3) duplicate-rescale new heads. Implementation ~50 LOC, careful test."
    )


# ---------------------------------------------------------------------------
# V6 wrapper expansion (easier, mostly width-doubling on conv/linear)
# ---------------------------------------------------------------------------


def expand_v6_wrapper(
    pico_state: dict[str, torch.Tensor],
    standard_model: V6Model,
) -> None:
    """Copy + width-expand the V6Model layers around the HAT backbone.

    Layers handled here (all relatively easy):
      - canvas_to_token: Linear(token_dim_old, token_dim_old) -> Linear(token_dim_new, token_dim_new)
        Both in + out widen. net2wider_linear handles directly.
      - fusion (PixelGaussianFusion): cross-attention with q/k/v/o linears.
        Internal num_heads + token_dim both change. Same head-change
        machinery as HAT.
      - pixel_head: Conv2d(feat_dim_old, feat_dim_old, 3) -> Conv2d(feat_dim_new, feat_dim_new, 3)
      - gaussian_spawner.conv: Conv2d(feat_dim_old, params_old, 1)
        params_old = 6 + token_dim_old. params_new = 6 + token_dim_new.
        net2wider_conv2d on both in and out.
      - rasterizer: non-parametric — no expansion.
      - composite_head Sequential of 3 Conv2d:
        Conv2d(feat_dim+token_dim, feat_dim, 3): both in + out widen
        Conv2d(feat_dim, hidden, 3): both widen
        Conv2d(hidden, 3, 3): in widens, out stays 3
        net2wider_conv2d cascaded; LAST layer's input is the rescaled side.

    KEY: the widened activations on each layer's OUTPUT propagate into the
    NEXT layer's INPUT. The next layer needs net2wider_conv2d/linear on
    its INPUT axis with the matching scale-down so the function is
    preserved.
    """
    raise NotImplementedError(
        "V6 wrapper expansion sketch above. Implementation ~100 LOC plus tests."
    )


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------


def expand_pico_to_standard(
    pico_ckpt_path: Path,
    output_path: Path,
    *,
    verify_eps: float = 1e-3,
) -> None:
    """End-to-end Pico -> Standard checkpoint expansion."""
    log.info("loading Pico ckpt from %s", pico_ckpt_path)
    pico_ckpt = torch.load(pico_ckpt_path, map_location="cpu", weights_only=False)
    pico_state = pico_ckpt.get("v6_model") or pico_ckpt.get("model_state_dict") or pico_ckpt

    # Reconstruct Pico for verification.
    pico_cfg = V6Config(backbone="hat-tiny", canvas_capacity=1500, token_dim=32,
                       cross_attention_heads=4)
    pico_model = V6Model(pico_cfg)
    pico_model.load_state_dict(pico_state, strict=False)
    pico_model.train(False)

    # Construct empty Standard.
    standard_cfg = V6Config(backbone="hat-small", canvas_capacity=5000, token_dim=64,
                           cross_attention_heads=6)
    standard_model = V6Model(standard_cfg)

    # Per-component expansion.
    expand_hat_tiny_to_small(
        {k: v for k, v in pico_state.items() if k.startswith("backbone.")},
        standard_model.backbone,
    )
    expand_v6_wrapper(pico_state, standard_model)

    # Verification: identical input -> identical output (within eps).
    torch.manual_seed(0)
    standard_model.train(False)
    lr = torch.randn(1, 9, 64, 64)
    standard_model.reset_state()
    pico_model.reset_state()
    with torch.no_grad():
        out_pico = pico_model(lr, motion_lr=None, frame_index=0)
        out_standard = standard_model(lr, motion_lr=None, frame_index=0)
    diff = (out_pico - out_standard).abs().mean().item()
    log.info("function-preservation check: |Pico - Standard| = %.6f (eps=%.6f)",
             diff, verify_eps)
    if diff > verify_eps:
        raise RuntimeError(
            f"function-preservation FAILED: |Pico - Standard|={diff:.6f} > eps={verify_eps}. "
            "Check per-layer rescaling — net2wider's scale-by-1/k rule must apply at every "
            "downstream consumer of a widened tensor."
        )

    # Save expanded ckpt.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    standard_state = standard_model.state_dict()
    standard_ckpt = {
        "step": 0,
        "kind": "v6_model",
        "v6_config": standard_cfg.__dict__,
        "v6_model": standard_state,
        "expanded_from": str(pico_ckpt_path),
        "expansion_method": "net2net_pico_to_standard_v1",
    }
    torch.save(standard_ckpt, output_path)
    log.info("expanded ckpt written to %s", output_path)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pico-ckpt", type=Path, required=True,
                   help="v6-Pico checkpoint to expand")
    p.add_argument("--output", type=Path, required=True,
                   help="Path to write the expanded Standard seed ckpt")
    p.add_argument("--verify-eps", type=float, default=1e-3,
                   help="Max acceptable |Pico - Standard| at expansion (function-preserving)")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")

    expand_pico_to_standard(
        pico_ckpt_path=args.pico_ckpt,
        output_path=args.output,
        verify_eps=args.verify_eps,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
