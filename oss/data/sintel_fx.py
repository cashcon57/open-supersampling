"""MPI Sintel dataset adapter for OSS-FX frame extrapolation training.

Holdout strategy: every 3rd frame (index % 3 == 0) is held out as pseudo-GT.
α is sampled uniformly from alpha_range per item; flow is used to warp
frames and construct the warped estimate and depth proxy.

Split: first 19 sequences alphabetically → train; last 4 → val.
"""
from __future__ import annotations

import struct
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

from oss.model.oss_fx import HISTORY_CH


def _read_flo(path: Path) -> Tensor:
    with open(path, "rb") as f:
        magic = struct.unpack("<f", f.read(4))[0]
        if abs(magic - 202021.25) > 0.001:
            raise ValueError(f"Invalid .flo magic {magic} in {path}")
        w, h = struct.unpack("<ii", f.read(8))
        data = torch.frombuffer(f.read(h * w * 2 * 4), dtype=torch.float32)
    return data.view(h, w, 2).permute(2, 0, 1).clone()  # (2, H, W)


def _warp_frame(frame: Tensor, flow: Tensor, alpha: float) -> Tensor:
    _, H, W = frame.shape
    scaled_flow = flow * alpha
    grid_y = (torch.arange(H, dtype=torch.float32, device=frame.device) + 0.5) * (2.0 / H) - 1.0
    grid_x = (torch.arange(W, dtype=torch.float32, device=frame.device) + 0.5) * (2.0 / W) - 1.0
    yy, xx = torch.meshgrid(grid_y, grid_x, indexing="ij")
    base = torch.stack([xx, yy], dim=-1).unsqueeze(0)
    disp = scaled_flow.permute(1, 2, 0).unsqueeze(0)
    disp[..., 0] = disp[..., 0] * (2.0 / W)
    disp[..., 1] = disp[..., 1] * (2.0 / H)
    grid = (base + disp).clamp(-2.0, 2.0)
    return F.grid_sample(
        frame.unsqueeze(0), grid, mode="bilinear", padding_mode="border", align_corners=False
    ).squeeze(0)


def _flow_magnitude_depth(flow: Tensor) -> Tensor:
    mag = flow.pow(2).sum(dim=0, keepdim=True).sqrt()
    max_val = mag.amax().clamp(min=1e-6)
    return (mag / max_val).clamp(0.0, 1.0)


def _load_png(path: Path) -> Tensor:
    from torchvision.io import read_image
    img = read_image(str(path)).float() / 255.0
    return img[:3]


_TRAIN_SEQUENCES = 19
_VAL_SEQUENCES = 4


def _split_sequences(all_seqs: list[str], split: str) -> list[str]:
    sorted_seqs = sorted(all_seqs)
    if split == "train":
        return sorted_seqs[:_TRAIN_SEQUENCES]
    return sorted_seqs[-_VAL_SEQUENCES:]


class SintelFxDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        alpha_range: tuple[float, float] = (0.1, 0.95),
        resolution: tuple[int, int] = (436, 1024),
        augment: bool = True,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.alpha_range = alpha_range
        self.resolution = resolution
        self.augment = augment

        pass_name = "clean" if split == "train" else "final"
        pass_dir = self.root / "training" / pass_name
        flow_dir = self.root / "training" / "flow"

        if not pass_dir.is_dir():
            raise FileNotFoundError(f"Sintel pass directory not found: {pass_dir}")

        all_seqs = [d.name for d in pass_dir.iterdir() if d.is_dir()]
        seq_names = _split_sequences(all_seqs, split)

        self._items: list[tuple[Path, Path, Path, Path, Path]] = []
        for seq in seq_names:
            frames = sorted((pass_dir / seq).glob("frame_*.png"))
            flows = sorted((flow_dir / seq).glob("frame_*.flo")) if (flow_dir / seq).is_dir() else []
            if len(frames) < 3 or len(flows) < 2:
                continue
            for i in range(0, len(frames) - 2):
                if (i + 2) % 3 != 0:
                    continue
                t_minus2 = frames[i]
                t_minus1 = frames[i + 1]
                t_gt = frames[i + 2]
                if i < len(flows) and i + 1 < len(flows):
                    flow_a = flows[i]
                    flow_b = flows[i + 1]
                    self._items.append((t_minus2, t_minus1, t_gt, flow_a, flow_b))

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        t_minus2_p, t_minus1_p, t_gt_p, flow_a_p, flow_b_p = self._items[idx]

        frame_tm2 = _load_png(t_minus2_p)
        frame_tm1 = _load_png(t_minus1_p)
        frame_gt = _load_png(t_gt_p)
        flow_a = _read_flo(flow_a_p)
        flow_b = _read_flo(flow_b_p)

        H, W = self.resolution
        if frame_tm1.shape[-2] != H or frame_tm1.shape[-1] != W:
            frame_tm2 = F.interpolate(frame_tm2.unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False).squeeze(0)
            frame_tm1 = F.interpolate(frame_tm1.unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False).squeeze(0)
            frame_gt = F.interpolate(frame_gt.unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False).squeeze(0)
            flow_a = F.interpolate(flow_a.unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False).squeeze(0)
            flow_b = F.interpolate(flow_b.unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False).squeeze(0)

        alpha_lo, alpha_hi = self.alpha_range
        alpha = alpha_lo + (alpha_hi - alpha_lo) * torch.rand(1).item() if self.augment else 0.5

        if self.augment:
            if torch.rand(1).item() > 0.5:
                frame_tm2 = TF.hflip(frame_tm2)
                frame_tm1 = TF.hflip(frame_tm1)
                frame_gt = TF.hflip(frame_gt)
                flow_a = TF.hflip(flow_a)
                flow_b = TF.hflip(flow_b)
                flow_a[0] = -flow_a[0]
                flow_b[0] = -flow_b[0]

            if torch.rand(1).item() > 0.5:
                frame_tm2, frame_tm1 = frame_tm1, frame_tm2
                flow_a = -flow_b
                flow_b = -flow_a.clone()

            hue_shift = (torch.rand(1).item() - 0.5) * 0.1
            sat_shift = 1.0 + (torch.rand(1).item() - 0.5) * 0.2
            frame_tm1 = TF.adjust_hue(frame_tm1, hue_shift)
            frame_tm1 = TF.adjust_saturation(frame_tm1, sat_shift)
            frame_gt = TF.adjust_hue(frame_gt, hue_shift)
            frame_gt = TF.adjust_saturation(frame_gt, sat_shift)

        warped = _warp_frame(frame_tm1, flow_b, alpha)
        target = (1.0 - alpha) * frame_tm1 + alpha * frame_gt
        depth = _flow_magnitude_depth(flow_b)
        history = torch.zeros(HISTORY_CH, H, W, dtype=torch.float32)

        return {
            "warped": warped,
            "depth": depth,
            "history": history,
            "alpha": torch.tensor(alpha, dtype=torch.float32),
            "target": target,
            "frame_t": frame_tm1,
        }
