"""Cosine LR with warm restarts for v6 training.

Per v6 memo §6: ``2e-4 cosine + 3 warm restarts, T_0=50_000, T_mult=1``,
``max_steps=300_000``. With T_mult=1, 3 restarts means 4 cycles × 50K =
200K total — the last 100K of training would run at LR=0. Pre-launch
audit caught this. Default ``num_restarts`` here is bumped to 5 so the
schedule covers 6 cycles × 50K = 300K. Callers running shorter recipes
should pass ``num_restarts`` explicitly to match their max_steps.

Why hand-roll this instead of using ``torch.optim.lr_scheduler.CosineAnnealingWarmRestarts``?
The torch built-in counts in epochs and restarts on hitting ``T_cur``; we
want a step-driven, deterministic schedule that we drive with an explicit
``step`` integer (matching how the rest of the v6 trainer logs).
"""
from __future__ import annotations

import math

import torch


class CosineLRWithWarmRestarts:
    """Cosine annealing with N warm restarts, step-indexed.

    Schedule:
      cycle 0: steps [0, T_0)            ->   lr cosines from base_lr to 0
      cycle 1: steps [T_0, T_0 + T_1)    ->   restart, cosine again, T_1 = T_0 * T_mult
      ...
      after the (num_restarts)-th restart finishes, lr stays at 0.

    Args:
        optimizer: Any ``torch.optim.Optimizer`` — we mutate ``param_group['lr']``
            on every ``step``.
        base_lr: Peak LR at the start of each cycle.
        T_0: Length of the first cycle in steps.
        T_mult: Multiplier applied to the cycle length at each restart.
            ``T_mult=1`` keeps the cycle length constant (the v6 default).
        num_restarts: Number of restarts after the initial cycle. ``num_restarts=3``
            means 4 cycles total.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        base_lr: float,
        T_0: int = 50_000,
        T_mult: float = 1.0,
        num_restarts: int = 5,
    ):
        if T_0 < 1:
            raise ValueError(f"T_0 must be >= 1, got {T_0}")
        if T_mult <= 0.0:
            raise ValueError(f"T_mult must be > 0, got {T_mult}")
        if num_restarts < 0:
            raise ValueError(f"num_restarts must be >= 0, got {num_restarts}")

        self.optimizer = optimizer
        self.base_lr = float(base_lr)
        self.T_0 = int(T_0)
        self.T_mult = float(T_mult)
        self.num_restarts = int(num_restarts)
        self._last_lr: float = self.base_lr

        # Precompute cycle boundaries for O(log) lookup if num_restarts ever
        # gets huge — for the v6 default of 3 a linear scan would be fine,
        # but we precompute anyway for clarity.
        self._cycle_lengths: list[int] = []
        self._cycle_starts: list[int] = []
        cur_start = 0
        cur_len = self.T_0
        for _ in range(self.num_restarts + 1):
            self._cycle_starts.append(cur_start)
            self._cycle_lengths.append(max(1, int(round(cur_len))))
            cur_start += self._cycle_lengths[-1]
            cur_len = cur_len * self.T_mult
        # End of the final cycle — anything past this stays at lr=0.
        self._end_step = cur_start

    def _lr_at(self, step: int) -> float:
        if step < 0:
            return self.base_lr
        if step >= self._end_step:
            return 0.0
        # Find which cycle ``step`` lives in (linear scan; trivial cost).
        cycle = 0
        while (
            cycle + 1 < len(self._cycle_starts)
            and step >= self._cycle_starts[cycle + 1]
        ):
            cycle += 1
        cycle_start = self._cycle_starts[cycle]
        cycle_len = self._cycle_lengths[cycle]
        t_in_cycle = step - cycle_start
        # Cosine from base_lr -> 0 over [0, cycle_len). At restart boundary
        # (step == cycle_start of the NEXT cycle) the previous cosine has
        # not yet hit zero; we restart cleanly anyway.
        progress = t_in_cycle / cycle_len
        return 0.5 * self.base_lr * (1.0 + math.cos(math.pi * progress))

    def step(self, step: int) -> None:
        """Set ``param_group['lr']`` on the optimizer for the given training step."""
        lr = self._lr_at(int(step))
        self._last_lr = lr
        for group in self.optimizer.param_groups:
            group["lr"] = lr

    def get_last_lr(self) -> float:
        """Return the LR set by the most recent ``step`` call."""
        return self._last_lr


__all__ = ["CosineLRWithWarmRestarts"]
