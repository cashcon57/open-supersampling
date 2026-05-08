#!/usr/bin/env python
"""Microbenchmark v6 ConcatFusion against PixelGaussianFusion.

Default synthetic case:
  - B=2, feat_dim=180, H=540, W=960
  - ConcatFusion fields at HR: G/m/I_base/depth/MV
  - PixelGaussianFusion Gaussian tokens: K=1024, token_dim=64

Timings use CUDA events over eager forward passes. If PyTorch or CUDA is not
available, the script prints an explicit NOT RUN result and exits without
fabricating timings.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

try:
    import torch
except Exception as exc:  # pragma: no cover - environment-dependent.
    torch = None  # type: ignore[assignment]
    TORCH_IMPORT_ERROR: Exception | None = exc
else:
    TORCH_IMPORT_ERROR = None


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@dataclass(frozen=True)
class BenchCase:
    module: str
    dtype: str
    status: str
    median_ms: float | None = None
    mean_ms: float | None = None
    p95_ms: float | None = None
    p99_ms: float | None = None
    min_ms: float | None = None
    max_ms: float | None = None
    peak_mib: float | None = None
    error: str | None = None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations.")
    parser.add_argument("--iters", type=int, default=100, help="Timed iterations.")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--feat-dim", type=int, default=180)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--latent-r", type=int, default=4)
    parser.add_argument("--concat-hidden", type=int, default=64)
    parser.add_argument("--gaussian-tokens", type=int, default=1024)
    parser.add_argument("--token-dim", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=6)
    parser.add_argument("--window-size", type=int, default=16)
    parser.add_argument(
        "--dtype",
        choices=["all", "fp32", "fp16"],
        default="all",
        help="Precision variant(s) to run.",
    )
    parser.add_argument(
        "--allow-tf32",
        action="store_true",
        help="Allow TF32 kernels for float32 matmul/convolution.",
    )
    return parser.parse_args(argv)


def percentile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("cannot percentile empty sample")
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def import_modules() -> tuple[type[torch.nn.Module], type[torch.nn.Module]]:
    try:
        from oss.sr.v6.concat_fusion import ConcatFusion
        from oss.sr.v6.cross_attention import PixelGaussianFusion
    except Exception as exc:  # pragma: no cover - environment-dependent message path.
        raise RuntimeError(f"could not import fusion modules: {exc}") from exc
    return ConcatFusion, PixelGaussianFusion


def summarize_timings(
    module: str,
    dtype_name: str,
    timings_ms: list[float],
    peak_mib: float,
) -> BenchCase:
    return BenchCase(
        module=module,
        dtype=dtype_name,
        status="OK",
        median_ms=statistics.median(timings_ms),
        mean_ms=statistics.fmean(timings_ms),
        p95_ms=percentile(timings_ms, 0.95),
        p99_ms=percentile(timings_ms, 0.99),
        min_ms=min(timings_ms),
        max_ms=max(timings_ms),
        peak_mib=peak_mib,
    )


def benchmark_forward(
    module_name: str,
    dtype_name: str,
    fn: Callable[[], torch.Tensor],
    warmup: int,
    iters: int,
    device: torch.device,
) -> BenchCase:
    if warmup < 0:
        raise ValueError("--warmup must be >= 0")
    if iters < 1:
        raise ValueError("--iters must be >= 1")

    timings_ms: list[float] = []
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    try:
        with torch.inference_mode():
            for _ in range(warmup):
                _ = fn()
            torch.cuda.synchronize(device)

            for _ in range(iters):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                _ = fn()
                end.record()
                end.synchronize()
                timings_ms.append(float(start.elapsed_time(end)))

        peak_mib = torch.cuda.max_memory_allocated(device) / 1024**2
        return summarize_timings(module_name, dtype_name, timings_ms, peak_mib)
    except RuntimeError as exc:
        torch.cuda.synchronize(device)
        return BenchCase(
            module=module_name,
            dtype=dtype_name,
            status="ERROR",
            error=f"{type(exc).__name__}: {exc}",
        )


def make_concat_case(
    args: argparse.Namespace,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.nn.Module, Callable[[], torch.Tensor]]:
    ConcatFusion, _ = import_modules()
    model = ConcatFusion(
        feat_dim=args.feat_dim,
        latent_R=args.latent_r,
        hidden=args.concat_hidden,
    ).to(device=device, dtype=dtype)
    model.eval()

    shape = (args.batch_size, args.feat_dim, args.height, args.width)
    latent_shape = (args.batch_size, args.latent_r, args.height, args.width)
    scalar_shape = (args.batch_size, 1, args.height, args.width)
    image_shape = (args.batch_size, 3, args.height, args.width)
    motion_shape = (args.batch_size, 2, args.height, args.width)

    F = torch.randn(shape, device=device, dtype=dtype)
    G = torch.randn(latent_shape, device=device, dtype=dtype)
    m = torch.randn(scalar_shape, device=device, dtype=dtype)
    I_base = torch.randn(image_shape, device=device, dtype=dtype)
    depth = torch.randn(scalar_shape, device=device, dtype=dtype)
    MV = torch.randn(motion_shape, device=device, dtype=dtype)

    def run() -> torch.Tensor:
        return model(F, G, m, I_base, depth, MV)

    return model, run


def make_pixel_gaussian_case(
    args: argparse.Namespace,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.nn.Module, Callable[[], torch.Tensor]]:
    _, PixelGaussianFusion = import_modules()
    model = PixelGaussianFusion(
        feat_dim=args.feat_dim,
        token_dim=args.token_dim,
        num_heads=args.num_heads,
        window_size=args.window_size,
    ).to(device=device, dtype=dtype)
    model.eval()

    pixel_features = torch.randn(
        (args.batch_size, args.feat_dim, args.height, args.width),
        device=device,
        dtype=dtype,
    )
    gaussian_tokens = torch.randn(
        (args.batch_size, args.gaussian_tokens, args.token_dim),
        device=device,
        dtype=dtype,
    )

    def run() -> torch.Tensor:
        return model(pixel_features, gaussian_tokens)

    return model, run


def dtype_variants(args: argparse.Namespace) -> list[tuple[str, torch.dtype]]:
    variants: list[tuple[str, torch.dtype]] = []
    if args.dtype in ("all", "fp32"):
        variants.append(("fp32", torch.float32))
    if args.dtype in ("all", "fp16"):
        variants.append(("fp16", torch.float16))
    return variants


def print_table(results: list[BenchCase]) -> None:
    print()
    print(
        f"{'module':<22} {'dtype':<6} {'status':<7} "
        f"{'median':>10} {'mean':>10} {'p95':>10} {'p99':>10} {'peak MiB':>10}"
    )
    print("-" * 96)
    for r in results:
        if r.status == "OK":
            print(
                f"{r.module:<22} {r.dtype:<6} {r.status:<7} "
                f"{r.median_ms:>9.3f} {r.mean_ms:>9.3f} "
                f"{r.p95_ms:>9.3f} {r.p99_ms:>9.3f} {r.peak_mib:>9.1f}"
            )
        else:
            print(f"{r.module:<22} {r.dtype:<6} {r.status:<7} {r.error}")


def print_verdict(results: list[BenchCase]) -> int:
    exit_code = 0
    by_dtype: dict[str, dict[str, BenchCase]] = {}
    for result in results:
        by_dtype.setdefault(result.dtype, {})[result.module] = result

    print()
    for dtype_name in sorted(by_dtype):
        pair = by_dtype[dtype_name]
        concat = pair.get("ConcatFusion")
        pixel = pair.get("PixelGaussianFusion")
        if concat is None or pixel is None:
            continue
        if concat.status != "OK" or pixel.status != "OK":
            print(f"VERDICT {dtype_name}: NOT RUN (one or both modules did not complete)")
            continue
        assert concat.median_ms is not None
        assert pixel.median_ms is not None
        if concat.median_ms <= pixel.median_ms:
            print(
                f"VERDICT {dtype_name}: PASS "
                f"(ConcatFusion {concat.median_ms:.3f} ms <= "
                f"PixelGaussianFusion {pixel.median_ms:.3f} ms median)"
            )
        else:
            print(
                f"VERDICT {dtype_name}: FAIL "
                f"(ConcatFusion {concat.median_ms:.3f} ms > "
                f"PixelGaussianFusion {pixel.median_ms:.3f} ms median)"
            )
            print("GitHub issue required for this acceptance failure.")
            exit_code = max(exit_code, 1)
    return exit_code


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if torch is None:
        print(f"NOT RUN: PyTorch is unavailable: {TORCH_IMPORT_ERROR}")
        return 0
    if not torch.cuda.is_available():
        print("NOT RUN: CUDA is unavailable; this benchmark requires CUDA events.")
        return 0

    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = bool(args.allow_tf32)
    torch.backends.cudnn.allow_tf32 = bool(args.allow_tf32)

    try:
        import_modules()
    except RuntimeError as exc:
        print(f"NOT RUN: {exc}")
        return 0

    print("Benchmarking v6 fusion modules")
    print(
        "shape="
        f"B{args.batch_size} C{args.feat_dim} H{args.height} W{args.width}; "
        f"concat latent_R={args.latent_r}; "
        f"pixel gaussian K={args.gaussian_tokens} token_dim={args.token_dim}; "
        f"warmup={args.warmup} iters={args.iters}"
    )
    print(f"device={torch.cuda.get_device_name(device)}")
    print(f"torch={torch.__version__} cuda_runtime={torch.version.cuda} tf32={args.allow_tf32}")

    results: list[BenchCase] = []
    for dtype_name, dtype in dtype_variants(args):
        for module_name, factory in (
            ("ConcatFusion", make_concat_case),
            ("PixelGaussianFusion", make_pixel_gaussian_case),
        ):
            model: torch.nn.Module | None = None
            fn: Callable[[], torch.Tensor] | None = None
            try:
                model, fn = factory(args, dtype, device)
                result = benchmark_forward(
                    module_name=module_name,
                    dtype_name=dtype_name,
                    fn=fn,
                    warmup=args.warmup,
                    iters=args.iters,
                    device=device,
                )
            except RuntimeError as exc:
                result = BenchCase(
                    module=module_name,
                    dtype=dtype_name,
                    status="ERROR",
                    error=f"{type(exc).__name__}: {exc}",
                )
            results.append(result)
            del fn
            del model
            torch.cuda.empty_cache()

    print_table(results)
    print()
    print("JSON_RESULTS " + json.dumps([asdict(r) for r in results], sort_keys=True))
    return print_verdict(results)


if __name__ == "__main__":
    raise SystemExit(main())
