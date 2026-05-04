"""GaussianField — SoA persistent state for the v5 Gaussian temporal track.

Storage layout (one row per Gaussian, capacity N):
    mu         : (N, 2)   pixel-space sub-pixel positions (x, y)
    log_scale  : (N, 2)   per-axis log-scale; scale = exp(log_scale)
    rotation   : (N,)     orientation in radians
    color      : (N, 3)   RGB
    opacity    : (N,)     alpha in [0, 1] post-sigmoid
    alive      : (N,)     bool — false rows are free slots ready for densification

History: a deque of up to 5 prior `GaussianField` snapshots (newest first).
Used by the multi-frame transformer to attend over previous fields.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque

import torch


HISTORY_LEN = 5


class GaussianField:
    def __init__(self, capacity: int = 16384, device: str | torch.device = "cpu") -> None:
        self.capacity = int(capacity)
        self.device = torch.device(device)
        self.mu = torch.zeros((self.capacity, 2), device=self.device)
        self.log_scale = torch.zeros((self.capacity, 2), device=self.device)
        self.rotation = torch.zeros((self.capacity,), device=self.device)
        self.color = torch.zeros((self.capacity, 3), device=self.device)
        self.opacity = torch.zeros((self.capacity,), device=self.device)
        self.alive = torch.zeros((self.capacity,), dtype=torch.bool, device=self.device)
        self._history: Deque["GaussianField"] = deque(maxlen=HISTORY_LEN)

    # ---- access -----------------------------------------------------------

    @property
    def history(self) -> list["GaussianField"]:
        return list(self._history)

    def count_alive(self) -> int:
        return int(self.alive.sum().item())

    # ---- mutators ---------------------------------------------------------

    def push_history(self, snapshot: "GaussianField") -> None:
        self._history.appendleft(snapshot)

    def to(self, device: str | torch.device) -> "GaussianField":
        device = torch.device(device)
        moved = GaussianField(capacity=self.capacity, device=device)
        moved.mu = self.mu.to(device)
        moved.log_scale = self.log_scale.to(device)
        moved.rotation = self.rotation.to(device)
        moved.color = self.color.to(device)
        moved.opacity = self.opacity.to(device)
        moved.alive = self.alive.to(device)
        # Move history snapshots too.
        moved._history = deque(
            (h.to(device) for h in self._history), maxlen=HISTORY_LEN
        )
        return moved

    def clone(self) -> "GaussianField":
        c = GaussianField(capacity=self.capacity, device=self.device)
        c.mu = self.mu.clone()
        c.log_scale = self.log_scale.clone()
        c.rotation = self.rotation.clone()
        c.color = self.color.clone()
        c.opacity = self.opacity.clone()
        c.alive = self.alive.clone()
        return c


__all__ = ["GaussianField", "HISTORY_LEN"]
