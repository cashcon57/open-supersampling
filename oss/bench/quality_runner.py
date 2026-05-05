from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from oss.bench.fsr1_reference import fsr1_upscale
from oss.valuation.metrics import lpips_dist, psnr, ssim


def _to_nchw(x) -> torch.Tensor:
    t = x if isinstance(x, torch.Tensor) else torch.from_numpy(np.asarray(x))
    t = t.float()
    if t.ndim == 3 and t.shape[-1] in (1, 3):
        t = t.permute(2, 0, 1)
    if t.ndim == 3:
        t = t.unsqueeze(0)
    return t


def _ldr(x: torch.Tensor) -> torch.Tensor:
    return (x / (x + 1.0)).clamp(0, 1)


class QualityRunner:
    def __init__(self, scale_factor: float = 2.0, device: str = "cpu", ckpt_path: str | None = None):
        self.scale_factor = float(scale_factor)
        self.device = torch.device(device)
        self.ckpt_path = ckpt_path
        self._pico = None

    def _load_pico(self):
        if self._pico is not None or not self.ckpt_path:
            return self._pico
        from oss.model.oss_pico import OSSPico

        state = torch.load(Path(self.ckpt_path), map_location=self.device)
        model = OSSPico().to(self.device).train(False)
        model.load_state_dict(state["model"])
        self._pico = model
        return self._pico

    def _score(self, pred: torch.Tensor, gt: torch.Tensor, ms: float) -> dict:
        pred = pred.cpu()
        gt = gt.cpu()
        pred_ldr, gt_ldr = _ldr(pred), _ldr(gt)
        return {
            "rgb_hr": pred,
            "psnr": float(psnr(pred, gt)),
            "ssim": float(ssim(pred_ldr, gt_ldr)),
            "lpips": float(lpips_dist(pred_ldr * 2 - 1, gt_ldr * 2 - 1)),
            "ms_per_frame": ms,
        }

    def _run_pico(self, lr: torch.Tensor, gt: torch.Tensor) -> torch.Tensor | None:
        model = self._load_pico()
        if model is None:
            return None
        b, _, h, w = lr.shape
        hh, hw = gt.shape[-2:]
        with torch.no_grad():
            rgb, _ = model(
                lr.to(self.device),
                torch.zeros(b, 1, h, w, device=self.device),
                torch.zeros(b, 2, h, w, device=self.device),
                torch.zeros(b, 3, h, w, device=self.device),
                torch.ones(b, 3, h, w, device=self.device),
                torch.zeros(b, 3, hh, hw, device=self.device),
                torch.zeros(b, model.HIDDEN_CHANNELS, h // 4, w // 4, device=self.device),
            )
        return rgb.cpu()

    def run_methods(self, lr_image, gt_hr_image) -> dict[str, dict]:
        lr = _to_nchw(lr_image)
        gt = _to_nchw(gt_hr_image)
        size = gt.shape[-2:]
        out = {}

        def run(name: str, fn):
            t0 = time.perf_counter()
            pred = fn()
            out[name] = self._score(pred, gt, (time.perf_counter() - t0) * 1000.0)

        run("bilinear", lambda: F.interpolate(lr, size=size, mode="bilinear", align_corners=False))
        run("bicubic", lambda: F.interpolate(lr, size=size, mode="bicubic", align_corners=False))
        run(
            "fsr1",
            lambda: torch.from_numpy(
                fsr1_upscale(lr[0].permute(1, 2, 0).numpy(), self.scale_factor)
            ).permute(2, 0, 1).unsqueeze(0),
        )
        if self.ckpt_path:
            run("ors_pico", lambda: self._run_pico(lr, gt))
        return out
