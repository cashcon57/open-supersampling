"""Exponential moving average wrapper for v6 teacher training.

EMA decouples the sampled weights (used for evaluation / inference / FID
scoring during training) from the noisy SGD trajectory. Per the v6 memo §6
the teacher uses ``beta=0.999``; students don't EMA.

Design
------
- EMA params live in a separate set of buffers — NOT registered as
  parameters of the source model — so they don't get duplicated by DDP and
  don't show up in optimizer state. ``state_dict()`` exposes them for
  checkpointing.
- ``update`` does ``ema = decay * ema + (1 - decay) * source.detach()`` over
  every parameter that ``requires_grad`` in the source. Buffers (e.g.
  batchnorm running stats) are copied straight from the source — averaging
  them tends to lag and degrade quality.
- ``swap_into`` is a context manager that temporarily swaps the EMA params
  into the source model so the same model object can run inference at the
  EMA weights, then restores the live SGD weights on exit.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch
import torch.nn as nn


class EMAModel:
    """Standard parameter EMA with deterministic update.

    Args:
        model: The source ``nn.Module``. EMA params are initialized from
            this model's current parameter values (cloned, detached).
        decay: EMA decay rate; default 0.999 per v6 memo §6.

    The wrapper itself is **not** an ``nn.Module`` — its tensors live
    outside the module graph so DDP / FSDP don't try to all-reduce them
    and the optimizer doesn't see them as trainable.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        if not (0.0 < decay < 1.0):
            raise ValueError(f"decay must be in (0, 1), got {decay}")
        self.decay = float(decay)
        # Snapshot only parameters that require_grad — frozen layers stay frozen
        # and don't need EMA. We also snapshot all buffers (so loading the EMA
        # state into a model gives a fully runnable object).
        self.shadow_params: dict[str, torch.Tensor] = {
            name: p.detach().clone()
            for name, p in model.named_parameters()
            if p.requires_grad
        }
        self.shadow_buffers: dict[str, torch.Tensor] = {
            name: b.detach().clone() for name, b in model.named_buffers()
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """In-place EMA update. ``ema = decay * ema + (1 - decay) * source``.

        Buffers are copied straight from the source (no averaging) — this
        matches torchvision / Diffusers / k-diffusion convention.
        """
        d = self.decay
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            shadow = self.shadow_params.get(name)
            if shadow is None:
                # New parameter (e.g. lazy module): seed from source.
                self.shadow_params[name] = p.detach().clone()
                continue
            # Use lerp to keep the math identical regardless of dtype.
            # ema.lerp_(p, 1 - d)  is  ema += (1 - d) * (p - ema)  is
            # ema = d * ema + (1 - d) * p  for any dtype.
            shadow.lerp_(p.detach().to(shadow.dtype).to(shadow.device), 1.0 - d)
        for name, b in model.named_buffers():
            self.shadow_buffers[name] = b.detach().clone()

    def state_dict(self) -> dict:
        """Return a checkpoint-able dict of EMA tensors and metadata."""
        return {
            "decay": self.decay,
            "shadow_params": {k: v.detach().clone() for k, v in self.shadow_params.items()},
            "shadow_buffers": {k: v.detach().clone() for k, v in self.shadow_buffers.items()},
        }

    def load_state_dict(self, state: dict) -> None:
        self.decay = float(state.get("decay", self.decay))
        self.shadow_params = {
            k: v.detach().clone() for k, v in state["shadow_params"].items()
        }
        self.shadow_buffers = {
            k: v.detach().clone() for k, v in state.get("shadow_buffers", {}).items()
        }

    @contextmanager
    def swap_into(self, model: nn.Module) -> Iterator[None]:
        """Temporarily install EMA weights into ``model`` for evaluation.

        On enter: stash the model's live params/buffers, copy in EMA values.
        On exit: restore the live values regardless of exception.
        """
        backup_params: dict[str, torch.Tensor] = {}
        backup_buffers: dict[str, torch.Tensor] = {}
        try:
            with torch.no_grad():
                for name, p in model.named_parameters():
                    if name in self.shadow_params:
                        backup_params[name] = p.detach().clone()
                        p.data.copy_(self.shadow_params[name].to(p.device, p.dtype))
                for name, b in model.named_buffers():
                    if name in self.shadow_buffers:
                        backup_buffers[name] = b.detach().clone()
                        b.data.copy_(self.shadow_buffers[name].to(b.device, b.dtype))
            yield
        finally:
            with torch.no_grad():
                for name, p in model.named_parameters():
                    if name in backup_params:
                        p.data.copy_(backup_params[name])
                for name, b in model.named_buffers():
                    if name in backup_buffers:
                        b.data.copy_(backup_buffers[name])


__all__ = ["EMAModel"]
