"""v6 composite training loss.

Implements the loss recipe from the v6 canonical memo §5
(``docs/superpowers/experiments/2026-05-05-v6-architecture-canonical.md``):

  Charbonnier (smooth L1)            weight 1.0
  LPIPS-VGG                          weight 1.0
  Multi-scale VGG-19 feature L1      weights {0.1, 0.1, 1.0, 1.0, 1.0}
                                     on relu1_1 / relu2_1 / relu3_1 / relu4_1 / relu5_1
  Wavelet L1 (haar, 2-level SWT)     weight 0.5
  GAN hinge (UNetD)                  weight 0.05, starts at step 20K
  Sobel edge L1                      weight 0.2
  Temporal consistency (L1)          weight 0.5 (when prev tensors provided)

The discriminator is implemented separately in ``discriminator.py``; here we
only consume its logits. ``temporal_consistency_loss`` is re-exported from
``oss.train.losses`` as a convenience.

Design notes
------------
- LPIPS and the multi-scale VGG features both walk a frozen torchvision VGG.
  We instantiate them in ``__init__`` and freeze (inference-mode +
  ``requires_grad_(False)``) to keep the discriminator path the only
  adversarial signal and keep the module graph DDP-safe before wrapping.
- The wavelet term reuses ``oss.model.wavelet.SWT2D`` which already implements
  shift-invariant à trous SWT with circular boundary conditions and is
  unit-tested for round-trip accuracy. We use ``haar`` here (not ``db2`` like
  the legacy ``oss/train/losses.py``) per the v6 spec — haar's two-tap
  high-pass is the cheap default for explicit high-frequency supervision.
- bf16 safety: reductions that square HDR-valued tensors run in fp32 with
  autocast disabled. The VGG / LPIPS modules are float32-only (torchvision
  pretrained), so we cast the inputs to float32 before forwarding through them.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from oss.train.losses import temporal_consistency_loss as _temporal_consistency_loss


log = logging.getLogger("oss.sr.v6.losses")


def _debug_nan_enabled() -> bool:
    return os.environ.get("OSS_V6_DEBUG_NAN", "").strip().lower() in {
        "1", "true", "yes", "on",
    }


def _tensor_stats(x: torch.Tensor) -> str:
    x_f = x.detach().float()
    if x_f.numel() == 0:
        return "mean=nan min=nan max=nan"
    return (
        f"mean={float(x_f.mean()):.9g} "
        f"min={float(x_f.amin()):.9g} "
        f"max={float(x_f.amax()):.9g}"
    )


def _component_value(x: torch.Tensor) -> float:
    x_f = x.detach().float()
    if x_f.numel() == 1:
        return float(x_f)
    return float(x_f.mean())


# Re-export so callers can grab everything from this module.
def temporal_consistency_loss(*args, **kwargs):
    """Re-exported from ``oss.train.losses`` — see that module for docs."""
    return _temporal_consistency_loss(*args, **kwargs)


# ---------------------------------------------------------------------------
# Charbonnier
# ---------------------------------------------------------------------------

def charbonnier_loss(
    pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-3
) -> torch.Tensor:
    """``mean( sqrt( (pred - target)^2 + eps^2 ) )`` — smooth L1.

    Drop-in replacement for L1 with smoother gradients near zero. Per the
    v6 memo §5: roughly +0.1-0.2 dB vs raw L1 on photoreal SR.
    """
    with torch.autocast(device_type=pred.device.type, enabled=False):
        diff = pred.float() - target.float()
        return torch.sqrt(diff.square() + eps * eps).mean()


# ---------------------------------------------------------------------------
# Sobel edge
# ---------------------------------------------------------------------------

# 3×3 Sobel filters as registered buffers via a tiny helper module.
_SOBEL_X = torch.tensor(
    [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
)
_SOBEL_Y = torch.tensor(
    [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]
)


def _sobel_grad_magnitude(x: torch.Tensor) -> torch.Tensor:
    """Per-channel Sobel gradient magnitude. Output same spatial size as ``x``."""
    B, C, H, W = x.shape
    x = x.float()
    sx = _SOBEL_X.to(device=x.device, dtype=x.dtype)
    sy = _SOBEL_Y.to(device=x.device, dtype=x.dtype)
    kx = sx[None, None, :, :].expand(C, 1, 3, 3).contiguous()
    ky = sy[None, None, :, :].expand(C, 1, 3, 3).contiguous()
    gx = F.conv2d(x, kx, padding=1, groups=C)
    gy = F.conv2d(x, ky, padding=1, groups=C)
    return torch.sqrt(gx * gx + gy * gy + 1e-12)


def sobel_edge_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """L1 between Sobel gradient magnitudes of ``pred`` and ``target``."""
    with torch.autocast(device_type=pred.device.type, enabled=False):
        return (
            _sobel_grad_magnitude(pred) - _sobel_grad_magnitude(target)
        ).abs().mean()


# ---------------------------------------------------------------------------
# Wavelet L1 (haar, 2 levels, high-frequency subbands only)
# ---------------------------------------------------------------------------

def wavelet_l1_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """L1 over high-frequency subbands of a haar 2-level SWT.

    Uses ``oss.model.wavelet.SWT2D`` (shift-invariant SWT with circular
    boundaries). Only the LH / HL / HH subbands are compared — the LL
    approximation is already supervised by the Charbonnier term, and the
    point of the wavelet term is explicit high-frequency supervision.
    """
    from oss.model.wavelet import SWT2D

    swt = SWT2D(levels=2, wavelet="haar").to(device=pred.device, dtype=pred.dtype)
    _, pred_details = swt(pred)
    with torch.no_grad():
        _, target_details = swt(target)
    loss = pred.new_zeros(())
    for j in range(len(pred_details)):
        for p, t in zip(pred_details[j], target_details[j]):
            loss = loss + (p - t).abs().mean()
    return loss


# ---------------------------------------------------------------------------
# Multi-scale VGG-19 feature L1
# ---------------------------------------------------------------------------

# torchvision VGG-19 layer indices in ``vgg19().features`` for the relu_X_1
# activations. Verified against ``torchvision.models.vgg.cfgs['E']``.
_VGG19_RELU_INDICES = {
    "relu1_1": 1,
    "relu2_1": 6,
    "relu3_1": 11,
    "relu4_1": 20,
    "relu5_1": 29,
}
_VGG_LAYER_WEIGHTS = (0.1, 0.1, 1.0, 1.0, 1.0)
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class MultiScaleVGGLoss(nn.Module):
    """VGG-19 feature L1 at relu_{1..5}_1 with the v6 weights.

    The VGG forward is run with grads disabled and the module flagged as
    not-training. Inputs are expected in [0, 1] (RGB) and are
    ImageNet-normalized internally to match what the pretrained VGG was
    trained on. Inputs are cast to float32 for the VGG forward (torchvision
    pretrained weights are float32) and the resulting scalar is cast back
    to the input dtype — keeps the loss bf16-safe end-to-end without paying
    for a half-precision VGG.
    """

    def __init__(self):
        super().__init__()
        from torchvision.models import VGG19_Weights, vgg19

        vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
        # We only need up to the deepest relu we sample (relu5_1 = idx 29).
        max_idx = max(_VGG19_RELU_INDICES.values())
        self.features = nn.Sequential(*list(vgg.children())[: max_idx + 1])
        self.features.float()
        self.features.train(False)
        for p in self.features.parameters():
            p.requires_grad_(False)

        self.register_buffer(
            "mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "std", torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1), persistent=False
        )

        self._layer_indices = list(_VGG19_RELU_INDICES.values())
        self._layer_weights = _VGG_LAYER_WEIGHTS

    def train(self, mode: bool = True):  # type: ignore[override]
        # The VGG submodule must always stay non-training; only this wrapper's
        # ``training`` flag toggles. Defensive against being put inside a
        # composite that flips ``train(True)``.
        super().train(mode)
        self.features.train(False)
        return self

    def _extract(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Forward through ``self.features`` capturing the relu_*_1 activations."""
        feats: list[torch.Tensor] = []
        wanted = set(self._layer_indices)
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in wanted:
                feats.append(x)
                if i == self._layer_indices[-1]:
                    break
        return feats

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=pred.device.type, enabled=False):
            in_dtype = pred.dtype
            # VGG pretrained weights are fp32; cast to fp32 for the forward.
            p = pred.float()
            t = target.float().detach()
            # Imagenet-normalize.
            p = (p - self.mean) / self.std
            t = (t - self.mean) / self.std

            with torch.no_grad():
                target_feats = self._extract(t)
            # Re-run pred forward with grads enabled.
            pred_feats = self._extract(p)

            loss = p.new_zeros(())
            for w, pf, tf in zip(self._layer_weights, pred_feats, target_feats):
                loss = loss + w * (pf - tf).abs().mean()
            return loss.to(in_dtype)


def multi_scale_vgg_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Functional wrapper that builds a fresh ``MultiScaleVGGLoss`` per call.

    Prefer instantiating ``MultiScaleVGGLoss`` once and reusing it (as
    ``V6CompositeLoss`` does) — this functional form is purely for tests and
    one-off invocations where caching the VGG isn't worth it.
    """
    return MultiScaleVGGLoss().to(pred.device)(pred, target)


# ---------------------------------------------------------------------------
# GAN hinge (Real-ESRGAN convention)
# ---------------------------------------------------------------------------

def gan_hinge_d_loss(
    real_logits: torch.Tensor, fake_logits: torch.Tensor
) -> torch.Tensor:
    """Hinge discriminator loss: ``E[relu(1 - D(real)) + relu(1 + D(fake))]``."""
    return F.relu(1.0 - real_logits).mean() + F.relu(1.0 + fake_logits).mean()


def gan_hinge_g_loss(fake_logits: torch.Tensor) -> torch.Tensor:
    """Hinge generator loss: ``-E[D(fake)]``."""
    return -fake_logits.mean()


# ---------------------------------------------------------------------------
# Composite
# ---------------------------------------------------------------------------

# v6 memo §5 weights.
_W_CHARBONNIER = 1.0
_W_LPIPS = 1.0
_W_VGG = 1.0  # the per-layer weights inside the multi-scale term carry the rest
_W_WAVELET = 0.5
_W_GAN = 0.05
_W_SOBEL = 0.2
_W_TEMPORAL = 0.5


class V6CompositeLoss(nn.Module):
    """Composite loss for v6 generator training (§5 of the canonical memo).

    Args:
        gan_warmup_until_step: GAN weight is 0 for ``step < gan_warmup_until_step``,
            then ``_W_GAN`` after. Default 20_000 per the memo.

    Forward signature::

        loss, parts = composite(pred, target, fake_logits, step,
                                pred_warped_prev=None, target_warped_prev=None)

    where ``parts`` is a ``dict[str, float]`` mapping component name to its
    pre-weight scalar (for dashboard plotting). Total loss is a weighted sum.
    """

    def __init__(
        self,
        gan_warmup_until_step: int = 20_000,
        use_lpips: bool = True,
        debug_nan: Optional[bool] = None,
    ):
        super().__init__()
        self.gan_warmup_until_step = int(gan_warmup_until_step)
        self.use_lpips = bool(use_lpips)
        self.debug_nan = _debug_nan_enabled() if debug_nan is None else bool(debug_nan)

        self._lpips: Optional[nn.Module] = None
        self.vgg: MultiScaleVGGLoss = MultiScaleVGGLoss()
        if self.use_lpips:
            self._init_lpips()

    def _record_component(
        self,
        parts: dict[str, Any],
        name: str,
        value: torch.Tensor,
        pred: torch.Tensor,
        target: torch.Tensor,
        step: int,
    ) -> None:
        if (
            not self.debug_nan
            or "first_non_finite_component" in parts
            or bool(torch.isfinite(value).all().detach().item())
        ):
            return
        parts["first_non_finite_component"] = name
        log.warning(
            "loss component non-finite: name=%s value=%s pred_stats=%s "
            "target_stats=%s step=%d",
            name,
            _component_value(value),
            _tensor_stats(pred),
            _tensor_stats(target),
            int(step),
        )

    def _init_lpips(self) -> None:
        if self._lpips is not None:
            return
        import lpips as _lpips_pkg
        net = _lpips_pkg.LPIPS(net="vgg", verbose=False)
        net.float()
        net.train(False)
        for p in net.parameters():
            p.requires_grad_(False)
        self._lpips = net

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        fake_logits: Optional[torch.Tensor],
        step: int,
        pred_prev: Optional[torch.Tensor] = None,
        motion_lr: Optional[torch.Tensor] = None,
        scale_factor: float = 2.0,
        pred_warped_prev: Optional[torch.Tensor] = None,
        target_warped_prev: Optional[torch.Tensor] = None,
        target_prev: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        in_dtype = pred.dtype
        parts: dict[str, Any] = {}

        # Charbonnier.
        l_char = charbonnier_loss(pred, target)
        parts["charbonnier"] = float(l_char.detach())
        self._record_component(parts, "charbonnier", l_char, pred, target, step)

        # Multi-scale VGG.
        l_vgg = self.vgg.to(pred.device)(pred.clamp(0.0, 1.0), target.clamp(0.0, 1.0))
        parts["vgg"] = float(l_vgg.detach())
        self._record_component(parts, "vgg", l_vgg, pred, target, step)

        if self.use_lpips:
            # LPIPS-VGG. lpips expects [-1, 1]; we operate on [0, 1] and remap.
            assert self._lpips is not None
            lpips_module = self._lpips.to(pred.device)
            # lpips weights are fp32 — cast and cast back for bf16 safety.
            with torch.autocast(device_type=pred.device.type, enabled=False):
                p32 = pred.float().clamp(0.0, 1.0) * 2.0 - 1.0
                t32 = target.float().clamp(0.0, 1.0).detach() * 2.0 - 1.0
                l_lpips = lpips_module(p32, t32).mean().to(in_dtype)
        else:
            l_lpips = pred.new_zeros(())
        parts["lpips"] = float(l_lpips.detach())
        self._record_component(parts, "lpips", l_lpips, pred, target, step)

        # Wavelet L1 (haar, 2 levels, HF subbands).
        l_wav = wavelet_l1_loss(pred, target)
        parts["wavelet"] = float(l_wav.detach())
        self._record_component(parts, "wavelet", l_wav, pred, target, step)

        # Sobel edge.
        l_sobel = sobel_edge_loss(pred, target)
        parts["sobel"] = float(l_sobel.detach())
        self._record_component(parts, "sobel", l_sobel, pred, target, step)

        # GAN hinge generator term, after warmup.
        if fake_logits is not None and step >= self.gan_warmup_until_step:
            l_gan = gan_hinge_g_loss(fake_logits)
            w_gan = _W_GAN
        else:
            l_gan = pred.new_zeros(())
            w_gan = 0.0
        parts["gan"] = float(l_gan.detach())
        self._record_component(parts, "gan", l_gan, pred, target, step)

        # Temporal consistency. Audit finding HIGH-H1: the bare form
        # |pred_t - warp(pred_prev)| penalizes ANY frame-to-frame change,
        # including correct change matched by GT motion. The paired form:
        #
        #   l_temp = | |warp(pred_prev) - pred_t| - |warp(target_prev) - target_t| |
        #
        # only fires when the prediction is MORE temporally inconsistent
        # than the target would be under the same motion warp. Fires when
        # the trainer threads target_prev OR pre-warped tensors. Falls
        # back to the bare form (logged separately) when only pred_prev
        # is available.
        if (
            pred_prev is not None
            and target_prev is not None
            and motion_lr is not None
        ):
            from oss.train.losses import warp_with_motion
            pred_prev_w = warp_with_motion(pred_prev, motion_lr, scale_factor)
            with torch.no_grad():
                target_prev_w = warp_with_motion(target_prev, motion_lr, scale_factor)
            pred_residual = (pred - pred_prev_w).abs()
            target_residual = (target - target_prev_w).abs()
            l_temp = (pred_residual - target_residual).abs().mean()
            w_temp = _W_TEMPORAL
        elif pred_warped_prev is not None and target_warped_prev is not None:
            pred_residual = (pred - pred_warped_prev).abs()
            target_residual = (target - target_warped_prev).abs().detach()
            l_temp = (pred_residual - target_residual).abs().mean()
            w_temp = _W_TEMPORAL
        elif pred_prev is not None and motion_lr is not None:
            # Bare form fallback — trainer didn't thread target_prev.
            l_temp = temporal_consistency_loss(
                pred, pred_prev, motion_lr, scale_factor=scale_factor,
            )
            w_temp = _W_TEMPORAL
            parts["temporal_unpaired"] = float(l_temp.detach())
        else:
            l_temp = pred.new_zeros(())
            w_temp = 0.0
        parts["temporal"] = float(l_temp.detach())
        self._record_component(parts, "temporal", l_temp, pred, target, step)

        total = (
            _W_CHARBONNIER * l_char
            + _W_VGG * l_vgg
            + _W_LPIPS * l_lpips
            + _W_WAVELET * l_wav
            + _W_SOBEL * l_sobel
            + w_gan * l_gan
            + w_temp * l_temp
        )
        parts["total"] = float(total.detach())
        return total, parts


__all__ = [
    "charbonnier_loss",
    "multi_scale_vgg_loss",
    "MultiScaleVGGLoss",
    "wavelet_l1_loss",
    "sobel_edge_loss",
    "gan_hinge_d_loss",
    "gan_hinge_g_loss",
    "temporal_consistency_loss",
    "V6CompositeLoss",
]
