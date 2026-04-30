"""Forward-pass latency benchmark."""
from __future__ import annotations
import time
import numpy as np
import torch


def bench_model(forward_fn, n_warmup: int = 20, n_iters: int = 100) -> dict:
    cuda = torch.cuda.is_available()
    for _ in range(n_warmup):
        _ = forward_fn()
    if cuda:
        torch.cuda.synchronize()
    times = []
    for _ in range(n_iters):
        if cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = forward_fn()
        if cuda:
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    return {
        "mean_ms": float(np.mean(times)),
        "p95_ms":  float(np.percentile(times, 95)),
        "min_ms":  float(np.min(times)),
    }
