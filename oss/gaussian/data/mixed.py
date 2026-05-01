"""Weighted multi-source mixer for OSS-Gaussian Sprint 4 training data.

Per the Sprint 4 plan (T4.3), training mixes four sources with the table:
    sintel    : 0.30
    tartanair : 0.50
    hypersim  : 0.15  (defaults shifted from cyberpunk while we wait on captures)
    srgd      : 0.05

This module owns the sampler ONLY — it doesn't depend on whether the on-disk
data is present. Callers pass already-constructed datasets (or None to skip).
"""

from __future__ import annotations

import random
from typing import Iterable, Mapping

from torch.utils.data import Dataset

from .base import GaussianDataset, GaussianTrainingExample


# Default training mix. Note: cyberpunk was deferred from the original plan;
# its weight (15%) has been redistributed onto hypersim (pretrain anchor).
DEFAULT_WEIGHTS: dict[str, float] = {
    "sintel": 0.30,
    "tartanair": 0.50,
    "hypersim": 0.15,
    "srgd": 0.05,
}


class MixedGaussianDataset(Dataset):
    """A length-N synthetic dataset that, on each ``__getitem__``, picks a
    sub-dataset by weighted sampling and proxies through to it.

    Args:
        datasets: mapping from source name → constructed ``GaussianDataset``.
            ``None`` values are skipped (so callers can build them lazily).
        weights: mapping from source name → relative weight. Re-normalized.
            Defaults to ``DEFAULT_WEIGHTS``.
        length: virtual epoch length; defaults to sum of sub-dataset sizes.
        seed: optional RNG seed for reproducible sampling.

    The combined index is virtual — the caller specifies how many samples per
    epoch and the mixer picks (source, sub-index) per access. Each sub-dataset
    is wrapped with cycling so ``length`` can exceed any individual size.
    """

    def __init__(
        self,
        datasets: Mapping[str, GaussianDataset | None],
        weights: Mapping[str, float] | None = None,
        length: int | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        # Drop any None entries.
        self._datasets: dict[str, GaussianDataset] = {
            k: v for k, v in datasets.items() if v is not None
        }
        if not self._datasets:
            raise ValueError(
                "MixedGaussianDataset requires at least one non-None sub-dataset"
            )

        # Compose final weight vector (only over sub-datasets we actually have).
        src_weights = dict(weights) if weights is not None else dict(DEFAULT_WEIGHTS)
        used = {k: src_weights[k] for k in self._datasets if k in src_weights}
        if not used:
            # Fall back to uniform if user passed unknown names.
            used = {k: 1.0 for k in self._datasets}
        total = sum(used.values())
        if total <= 0:
            raise ValueError(f"weights must sum to >0; got {used}")
        self._names = list(used.keys())
        self._weights = [used[k] / total for k in self._names]

        if length is None:
            length = sum(len(d) for d in self._datasets.values())
            if length <= 0:
                raise ValueError("All sub-datasets are empty")
        self._length = int(length)
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> GaussianTrainingExample:
        # Idx is a virtual cursor; we use it to seed a deterministic per-call
        # sub-pick when caller iterates (PyTorch DataLoader passes integers).
        # The randomness uses self._rng for source selection so different
        # epochs see different distributions.
        name = self._rng.choices(self._names, weights=self._weights, k=1)[0]
        ds = self._datasets[name]
        sub_idx = idx % len(ds)
        return ds[sub_idx]

    # ---- Diagnostics -------------------------------------------------------
    @property
    def weights(self) -> dict[str, float]:
        return {n: w for n, w in zip(self._names, self._weights)}

    def empirical_distribution(self, samples: int = 1000) -> dict[str, float]:
        """Sample-based estimate of the realized weight per source.

        Useful for the train run's startup log line and for unit tests.
        """
        counts = {n: 0 for n in self._names}
        local_rng = random.Random(self._rng.random())
        for _ in range(samples):
            name = local_rng.choices(self._names, weights=self._weights, k=1)[0]
            counts[name] += 1
        return {n: c / samples for n, c in counts.items()}

    def describe(self) -> str:
        parts = [f"{n}={w:.2f}(N={len(self._datasets[n])})" for n, w in zip(self._names, self._weights)]
        return f"MixedGaussianDataset[length={self._length}, " + ", ".join(parts) + "]"


__all__ = ["MixedGaussianDataset", "DEFAULT_WEIGHTS"]
