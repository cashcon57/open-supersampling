"""ORU-Pico - Steam Deck (RDNA 2) tier of the ORU upscaler family.

Pico is a temporal-recurrent kernel-prediction U-Net targeting the lowest
realistic shipping target for v0.2: an 8 CU RDNA 2 iGPU at 15 W with a 540p /
360p / 240p internal render pass upscaled 2x. The architecture is intentionally
conservative on params (~280 K) to leave headroom for the rest of the frame.

Architecture:
- Two-branch encoder: color+history vs G-buffer (depth/motion/normals), late
  fused at level 0 (MUNet I3D 2025 layout).
- 4-level U-Net at LR. Each level has a stage-entry (Down/Up) block and three
  refinement ConvBlocks at that scale. All channels are multiples of 8 for FP16
  packed math.
- Recurrent latent cell at the bottleneck (level 2, H/4) propagates a 24-ch
  hidden state across frames. Hidden state is None at sequence boundary.
- Kernel-prediction head at HR (Bako et al. 2017 / NPPD): predicts a softmaxed
  5x5 kernel per HR pixel applied to the bilinearly-upsampled noisy color so
  outputs are bounded convex combinations of neighbors.

Forward signature is fixed by the v0.2 ship contract; see docstring on `forward`.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ConvBlock, DownBlock, KernelPredictionHead, UpBlock
from .temporal import RecurrentLatentCell
from .wavelet import WaveletKPNHead


# Channel widths per the v0.2-pico design spec. All multiples of 8.
_PICO_CHANNELS = [16, 24, 32, 40]
# Hidden state channels for the recurrent bottleneck cell.
_PICO_HIDDEN_CHANNELS = 24
# Number of refinement ConvBlocks per spatial level (encoder + decoder).
# Three keeps the model in the 200-320K param window with the spec channels.
_REFINE_DEPTH = 3
# Default scale factor used at construction time when caller doesn't override.
# Steam Deck targets are 1.4× (540p→800p), 1.5×, 1.7×, 2.0×, 2.5× (320p→800p),
# 3.3× (240p→800p). The model accepts ANY positive float scale_factor at
# construction; train at the scale you'll deploy at, or train with multi-scale
# augmentation in the trainer.
_PICO_DEFAULT_SCALE_FACTOR = 2.0

# DLSS-style preset names mapped to scale_factor. Mirrors NVIDIA's published
# preset ratios — gives users a familiar dial. "DLAA" runs the model at
# native res (no upscale), pure temporal-AA + denoise.
PRESET_SCALE_FACTORS: dict[str, float] = {
    "DLAA":              1.0,    # no upscale; temporal AA + denoise only
    "Ultra Quality":     1.3,    # 75% input res
    "Quality":           1.5,    # 67% input res
    "Balanced":          1.7,    # 59% input res
    "Performance":       2.0,    # 50% input res
    "Ultra Performance": 3.0,    # 33% input res — extreme regime
}


def preset_scale(name: str) -> float:
    """Look up a DLSS-style preset name → scale factor.

    Accepts case-insensitive variants ('quality', 'Quality', 'QUALITY') and
    common shorthands ('ultra-perf', 'ultra_perf'). Raises KeyError for unknown
    names so callers can decide to error vs fall back to a default.
    """
    norm = name.strip().lower().replace("-", " ").replace("_", " ")
    aliases = {
        "dlaa": "DLAA",
        "ultra quality": "Ultra Quality",
        "uq": "Ultra Quality",
        "quality": "Quality",
        "q": "Quality",
        "balanced": "Balanced",
        "b": "Balanced",
        "performance": "Performance",
        "perf": "Performance",
        "p": "Performance",
        "ultra performance": "Ultra Performance",
        "ultra perf": "Ultra Performance",
        "up": "Ultra Performance",
    }
    if norm in aliases:
        return PRESET_SCALE_FACTORS[aliases[norm]]
    raise KeyError(
        f"unknown preset {name!r}; available: {list(PRESET_SCALE_FACTORS)}"
    )


def _refine_stack(ch: int, depth: int = _REFINE_DEPTH) -> nn.Sequential:
    """`depth` ConvBlocks at the same spatial level + channel count."""
    return nn.Sequential(*(ConvBlock(ch, ch) for _ in range(depth)))


class ORUPico(nn.Module):
    """Pico-tier ORU with radiance demodulation (NSRD Li 2024).

    Inputs (all expected float32):
        color_lr   : (B, 3, H_lr, W_lr) - noisy LR radiance (linear).
        depth_lr   : (B, 1, H_lr, W_lr) - LR depth.
        motion_lr  : (B, 2, H_lr, W_lr) - LR motion vectors.
        normals_lr : (B, 3, H_lr, W_lr) - LR normals (sh_normal style).
        albedo_lr  : (B, 3, H_lr, W_lr) - LR albedo (reflectance).
        history_hr : (B, 3, H_hr, W_hr) - prior frame's denoised HR output.
                                          Pass zeros at sequence boundary.
        hidden_state : (B, 24, H_lr/4, W_lr/4) | None - recurrent state.
                                          Pass None at sequence boundary.

    Outputs:
        rgb_hr : (B, 3, H_hr, W_hr) - denoised + upscaled RGB (re-modulated).
        new_hidden_state : (B, 24, H_lr/4, W_lr/4) - propagate to next frame.

    Radiance demodulation (NSRD):
    - Before network: color_demod = color_lr / (albedo_lr + epsilon)
      (network learns lighting, not material)
    - After network: rgb_hr = rgb_hr * albedo_hr_upsampled
      (re-modulate with upsampled HR albedo)
    """

    # Exposed for tests / external code that needs to allocate a zero hidden.
    HIDDEN_CHANNELS = _PICO_HIDDEN_CHANNELS
    SCALE_FACTOR = _PICO_DEFAULT_SCALE_FACTOR  # legacy alias

    def __init__(self, use_wavelet: bool = True, scale_factor: float = _PICO_DEFAULT_SCALE_FACTOR):
        """Construct an ORU-Pico network.

        Args:
            use_wavelet: When ``True`` (default, ship config), the final HR
                kernel-prediction head operates in stationary-wavelet space
                per Poudel 2025 (arXiv:2508.16024) — predicts a kernel for
                each of the 7 SWT subbands (1 LL + 2x3 details at db2 / 2
                levels), then inverse-SWT recombines into RGB. Per-paper
                expected gain: +1.5 dB PSNR / -17% LPIPS at <0.1 ms / 800p
                inference cost. ``False`` keeps the legacy single-RGB-kernel
                head — retained for ablation comparison only.
        """
        super().__init__()
        if scale_factor <= 0:
            raise ValueError(f"scale_factor must be positive; got {scale_factor}")
        self.use_wavelet = use_wavelet
        self.scale_factor = float(scale_factor)
        c = _PICO_CHANNELS

        # ---- Two-branch encoder, late-fused at level 0 ------------------
        # Branch A: noisy color (3) + warped history downsampled to LR (3).
        self.color_in = ConvBlock(3 + 3, c[0])
        # Branch B: G-buffer (depth 1 + motion 2 + normals 3 = 6).
        self.gbuf_in = ConvBlock(1 + 2 + 3, c[0])
        self.fuse = ConvBlock(c[0] * 2, c[0])
        self.enc0_refine = _refine_stack(c[0])

        # ---- Encoder downsamples ----------------------------------------
        self.enc1_down = DownBlock(c[0], c[1])
        self.enc1_refine = _refine_stack(c[1])
        self.enc2_down = DownBlock(c[1], c[2])
        self.enc2_refine = _refine_stack(c[2])
        self.enc3_down = DownBlock(c[2], c[3])
        self.enc3_refine = _refine_stack(c[3])

        # ---- Recurrent latent cell at bottleneck level 2 (H/4) ----------
        # The cell consumes / produces the 32-ch feature map at H/4 and threads
        # a 24-ch hidden state across frames.
        self.recur = RecurrentLatentCell(
            channels=c[2], hidden_channels=_PICO_HIDDEN_CHANNELS
        )

        # ---- Decoder upsamples (skip from same-level encoder) -----------
        self.dec3_up = UpBlock(c[3], c[2])
        self.dec3_refine = _refine_stack(c[2])
        self.dec2_up = UpBlock(c[2] * 2, c[1])  # +skip from x2_with_state
        self.dec2_refine = _refine_stack(c[1])
        self.dec1_up = UpBlock(c[1] * 2, c[0])  # +skip from x1
        self.dec1_refine = _refine_stack(c[0])

        # ---- Penultimate projection (no norm) ---------------------------
        # cat(d1, x0) at LR -> 32 ch features that feed the kernel head.
        self.penult = nn.Conv2d(c[0] * 2, 32, kernel_size=3, padding=1)

        # ---- Kernel-prediction head at HR -------------------------------
        # Wavelet-space head (Poudel 2025) is the default; legacy RGB-space
        # head retained behind use_wavelet=False for ablation comparison.
        if use_wavelet:
            # WaveletKPNHead works over the bilinearly-upsampled HR features +
            # noisy color, so the scale_factor here doesn't actually drive any
            # internal interpolation — kept for API symmetry. Pass the float
            # scale; the head accepts it for documentation purposes.
            self.kpn = WaveletKPNHead(
                feature_ch=32,
                kernel_size=5,
                scale_factor=self.scale_factor,
                levels=2,
                wavelet="db2",
            )
        else:
            self.kpn = KernelPredictionHead(32, kernel_size=5)

    def _hr_size(self, lr_h: int, lr_w: int) -> tuple[int, int]:
        """Compute HR spatial size from LR using this instance's scale_factor.

        Floor (not banker's round) for deterministic round-trip with the
        paired downsample step in the trainer. Same convention as the legacy
        scale_factor=2.0 path.
        """
        return (
            int(lr_h * self.scale_factor),
            int(lr_w * self.scale_factor),
        )

    def forward(
        self,
        color_lr: torch.Tensor,
        depth_lr: torch.Tensor,
        motion_lr: torch.Tensor,
        normals_lr: torch.Tensor,
        albedo_lr: torch.Tensor,
        history_hr: torch.Tensor,
        hidden_state: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, _, H_lr, W_lr = color_lr.shape

        # ---- Radiance demodulation (NSRD) ---------------------------------
        # Divide color by albedo to remove material; network learns lighting only.
        # Epsilon avoids division by zero in dark regions.
        color_demod = color_lr / (albedo_lr + 1e-3)

        # Downsample history HR -> LR for late-fused recurrent context.
        # External warping (motion-vector reproject) is the trainer's job; this
        # module just consumes whatever HR history was handed to it.
        history_lr = F.interpolate(
            history_hr, size=(H_lr, W_lr), mode="bilinear", align_corners=False
        )

        # ---- Encoder ---------------------------------------------------
        # Use demodulated color instead of raw color.
        rad = self.color_in(torch.cat([color_demod, history_lr], dim=1))
        # G-buffer unchanged: depth, motion, normals (no material dependence).
        gbuf = self.gbuf_in(torch.cat([depth_lr, motion_lr, normals_lr], dim=1))
        x0 = self.fuse(torch.cat([rad, gbuf], dim=1))
        x0 = self.enc0_refine(x0)

        x1 = self.enc1_refine(self.enc1_down(x0))
        x2 = self.enc2_refine(self.enc2_down(x1))
        x3 = self.enc3_refine(self.enc3_down(x2))

        # ---- Recurrent latent at bottleneck (level 2, H/4) --------------
        x2_with_state, new_hidden = self.recur(x2, hidden_state)

        # ---- Decoder ---------------------------------------------------
        d3 = self.dec3_refine(self.dec3_up(x3))
        d2 = self.dec2_refine(
            self.dec2_up(torch.cat([d3, x2_with_state], dim=1))
        )
        d1 = self.dec1_refine(self.dec1_up(torch.cat([d2, x1], dim=1)))

        # ---- Penultimate features at LR --------------------------------
        feats_lr = self.penult(torch.cat([d1, x0], dim=1))

        # ---- Bilinear upsample features and demodulated color to HR -----
        out_h, out_w = self._hr_size(H_lr, W_lr)
        feats_hr = F.interpolate(
            feats_lr, size=(out_h, out_w), mode="bilinear", align_corners=False
        )
        color_demod_hr_bilinear = F.interpolate(
            color_demod, size=(out_h, out_w), mode="bilinear", align_corners=False
        )

        # ---- Kernel-prediction head at HR (on demodulated lighting) ------
        rgb_demod_hr = self.kpn(feats_hr, color_demod_hr_bilinear)

        # ---- Radiance re-modulation (NSRD) --------------------------------
        # Upsample LR albedo to HR, then multiply back in.
        albedo_hr = F.interpolate(
            albedo_lr, size=(out_h, out_w), mode="bilinear", align_corners=False
        )
        rgb_hr = rgb_demod_hr * albedo_hr

        return rgb_hr, new_hidden
