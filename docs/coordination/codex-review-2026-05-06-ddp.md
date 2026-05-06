# Codex DDP Review - 2026-05-06

Scope: every `.py` under `oss/sr/v6/`, plus the requested `scripts/sr_train_v6.py` smoke entry point.

## Finding 1

- Severity: HIGH
- File and line range: `scripts/sr_train_v6.py:187-224`
- Description: The requested DDP smoke harness does not exercise the DDP parameter-sync path. In `--smoke`, the script constructs `V6Model`, writes `metrics.json`, and exits without wrapping the model in `torch.nn.parallel.DistributedDataParallel`, running a forward/backward, or checking gradients with `find_unused_parameters=True`. A green result from this smoke would only prove process-group setup plus model construction, not that parameters synchronize or that conditional-unused parameters are handled.
- Suggested fix: In `--smoke` under distributed launch, wrap `V6Model` in DDP with `find_unused_parameters=True`, run a tiny CPU forward/backward on `hat-tiny`, and include both the empty-canvas path (`K=0`, fusion/canvas projection unused) and a non-empty synthetic canvas path (`K>0`, fusion/canvas projection used).

## Finding 2

- Severity: HIGH
- File and line range: `scripts/sr_train_v6.py:122-124`
- Description: The exact requested smoke command fails CLI parsing once rendezvous succeeds because it passes `--no-bf16`, but the script only defines `--bf16` with `default=True`. Local parser check failed with `error: unrecognized arguments: --no-bf16`.
- Suggested fix: Add a paired boolean CLI using `argparse.BooleanOptionalAction`, e.g. `p.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)`, or add an explicit `--no-bf16` flag that sets `args.bf16 = False`.

## Finding 3

- Severity: HIGH
- File and line range: requested DDP smoke command / launcher environment
- Description: The exact requested command failed before worker launch. `torch.distributed.run --standalone --nproc_per_node=2 -- ... --no-bf16` repeatedly tried to connect to `localhost:0`, emitted `c10d` IPv6 lookup warnings, and timed out after two 60s attempts with `RendezvousConnectionError: The connection to the C10d store has failed`. An adjusted launch with `--master_addr=127.0.0.1 --master_port=29611` also failed in this sandbox with `DistNetworkError: The server socket has failed to bind ... EPERM`, so the requested smoke did not succeed here.
- Suggested fix: Make the smoke command deterministic for local/macOS runs by avoiding `--standalone` port `0` in restricted environments; pass an explicit loopback endpoint/port in CI or document the required host-network permission. After the launcher is fixed, keep `--no-bf16` in the command only after the CLI supports it.

## Finding 4

- Severity: MEDIUM
- File and line range: `oss/sr/v6/losses.py:278-317`
- Description: `V6CompositeLoss` lazily creates and assigns an LPIPS `nn.Module` inside `forward()`. If this loss module is ever wrapped, pickled, or replicated before first forward, its module graph changes later and can diverge by rank if one process takes an exception or skips the LPIPS path. Loss modules are usually not DDP-wrapped, but the task scope says every `nn.Module` under `oss/sr/v6/`, and DDP expects module parameters/buffers to be fixed after wrapping.
- Suggested fix: Instantiate LPIPS in `__init__` behind an explicit constructor option, or ensure the trainer eagerly calls `_init_lpips()` on every rank before any DDP wrapping/pickling boundary. Keep LPIPS parameters frozen.

## Notes

- No learnable parameter stored as a registered buffer was found in the inspected v6 modules. Learnable state is registered through `nn.Parameter` or child modules.
- `V6Model` has expected conditional-unused parameters: empty canvas returns `K=0`, so `canvas_to_token` and fusion parameters are unused on that forward. This is compatible with `find_unused_parameters=True`, but the current smoke does not verify it.
- No shared CUDA streams or custom CUDA memory pools were found in `oss/sr/v6/`.
