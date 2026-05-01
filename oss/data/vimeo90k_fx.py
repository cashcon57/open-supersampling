"""Vimeo-90K septuplet dataset adapter for OSS-FX frame extrapolation training.

Uses im3 and im5 as context frames, im4 as pseudo-GT at α=0.5.
Warped estimate: warp im3 toward im5 by alpha using their pixel difference
as a crude flow proxy. The SCN learns to correct the crude warp artifact.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

from oss.model.oss_fx import HISTORY_CH


def _load_png(path: Path) -> Tensor:
    from torchvision.io import read_image
    img = read_image(str(path)).float() / 255.0
    return img[:3]


def _warp_by_diff(frame_a: Tensor, frame_b: Tensor, alpha: float) -> Tensor:
    _, H, W = frame_a.shape
    lum_a = 0.2126 * frame_a[0] + 0.7152 * frame_a[1] + 0.0722 * frame_a[2]
    lum_b = 0.2126 * frame_b[0] + 0.7152 * frame_b[1] + 0.0722 * frame_b[2]
    diff = (lum_b - lum_a).unsqueeze(0).unsqueeze(0)
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=diff.dtype, device=diff.device).view(1, 1, 3, 3)
    sobel_y = sobel_x.transpose(-1, -2)
    grad_x = F.conv2d(diff, sobel_x, padding=1).squeeze()
    grad_y = F.conv2d(diff, sobel_y, padding=1).squeeze()
    flow = torch.stack([grad_x, grad_y], dim=0)

    scaled = flow * alpha
    grid_y = (torch.arange(H, dtype=torch.float32, device=frame_a.device) + 0.5) * (2.0 / H) - 1.0
    grid_x = (torch.arange(W, dtype=torch.float32, device=frame_a.device) + 0.5) * (2.0 / W) - 1.0
    yy, xx = torch.meshgrid(grid_y, grid_x, indexing="ij")
    base = torch.stack([xx, yy], dim=-1).unsqueeze(0)
    disp = scaled.permute(1, 2, 0).unsqueeze(0)
    disp[..., 0] = disp[..., 0] * (2.0 / W)
    disp[..., 1] = disp[..., 1] * (2.0 / H)
    grid = (base + disp).clamp(-2.0, 2.0)
    return F.grid_sample(
        frame_a.unsqueeze(0), grid, mode="bilinear", padding_mode="border", align_corners=False
    ).squeeze(0)


def _diff_magnitude_depth(frame_a: Tensor, frame_b: Tensor) -> Tensor:
    diff = (frame_b - frame_a).pow(2).sum(dim=0, keepdim=True).sqrt()
    max_val = diff.amax().clamp(min=1e-6)
    return (diff / max_val).clamp(0.0, 1.0)


class Vimeo90kFxDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        alpha_range: tuple[float, float] = (0.1, 0.95),
        resolution: tuple[int, int] = (256, 448),
        augment: bool = True,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.alpha_range = alpha_range
        self.resolution = resolution
        self.augment = augment

        list_file = "sep_trainlist.txt" if split == "train" else "sep_testlist.txt"
        list_path = self.root / list_file
        if not list_path.exists():
            raise FileNotFoundError(f"Vimeo-90K list file not found: {list_path}")

        seqs = list_path.read_text().splitlines()
        self._seq_dirs: list[Path] = []
        for rel in seqs:
            rel = rel.strip()
            if not rel:
                continue
            d = self.root / "sequences" / rel
            if (d / "im3.png").exists() and (d / "im4.png").exists() and (d / "im5.png").exists():
                self._seq_dirs.append(d)

    def __len__(self) -> int:
        return len(self._seq_dirs)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        seq_dir = self._seq_dirs[idx]

        frame3 = _load_png(seq_dir / "im3.png")
        frame4 = _load_png(seq_dir / "im4.png")
        frame5 = _load_png(seq_dir / "im5.png")

        H, W = self.resolution
        if frame3.shape[-2] != H or frame3.shape[-1] != W:
            frame3 = F.interpolate(frame3.unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False).squeeze(0)
            frame4 = F.interpolate(frame4.unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False).squeeze(0)
            frame5 = F.interpolate(frame5.unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False).squeeze(0)

        alpha_lo, alpha_hi = self.alpha_range
        alpha = alpha_lo + (alpha_hi - alpha_lo) * torch.rand(1).item() if self.augment else 0.5

        if self.augment:
            if torch.rand(1).item() > 0.5:
                frame3 = TF.hflip(frame3)
                frame4 = TF.hflip(frame4)
                frame5 = TF.hflip(frame5)

            if torch.rand(1).item() > 0.5:
                frame3 = TF.vflip(frame3)
                frame4 = TF.vflip(frame4)
                frame5 = TF.vflip(frame5)

            hue_shift = (torch.rand(1).item() - 0.5) * 0.1
            sat_shift = 1.0 + (torch.rand(1).item() - 0.5) * 0.2
            frame3 = TF.adjust_hue(frame3, hue_shift)
            frame3 = TF.adjust_saturation(frame3, sat_shift)
            frame5 = TF.adjust_hue(frame5, hue_shift)
            frame5 = TF.adjust_saturation(frame5, sat_shift)
            frame4 = TF.adjust_hue(frame4, hue_shift)
            frame4 = TF.adjust_saturation(frame4, sat_shift)

        warped = _warp_by_diff(frame3, frame5, alpha)
        depth = _diff_magnitude_depth(frame3, frame5)
        history = torch.zeros(HISTORY_CH, H, W, dtype=torch.float32)

        return {
            "warped": warped,
            "depth": depth,
            "history": history,
            "alpha": torch.tensor(alpha, dtype=torch.float32),
            "target": frame4,
            "frame_t": frame3,
        }
