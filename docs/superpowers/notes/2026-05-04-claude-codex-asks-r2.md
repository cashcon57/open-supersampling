# 2026-05-04 — Claude→Codex asks, round 2

Cash authorized me to keep assigning Codex independent verification + drafting work via the shared review surface. Round 1 (C1–C4) is fully discharged. Round 2 below — Cash said "all in parallel"; pick whichever is least-blocked.

Mark each item `claimed by Codex` / `done by Codex at HH:MM CDT` with a one-line note when handled. If a probe finds a real bug, file it under `## Open Findings` in `2026-05-04-v5-rolling-review.md` with severity + file:line citations.

## C5 — Vendor optimization audit notes draft

Severity: low (S6 prep, no impact on running training)

Cash flagged in the original handoff brief: "Vendor-optimization audit (CUDA mma.sync, AMD wave32 vs wave64, Apple TBDR-applies-to-compute-or-not) lives in conversation context — pin to a docs/superpowers/notes/ file when you hit S6 perf pass." The conversation transcript contained at least one corrected claim (Wave64-on-AMD-RDNA-2; TBDR-applies-to-compute) — the spec / repo are the ground truth, not the transcript.

Deliverable: `docs/superpowers/notes/vendor-optimization-audit.md` with:

- Per-vendor matrix of available matrix-acceleration paths (NVIDIA Ampere/Ada/Hopper tensor cores via `wmma`/`mma.sync`/TMA; AMD CDNA MFMA + RDNA 3 WMMA; Intel XMX via Level Zero; Apple ANE/AMX). Cite the official ISA / driver doc for each, not transcript memory.
- For each: minimum precision (FP16 / BF16 / INT8), cooperative-thread requirements (warp size 32 vs wave 32/64), and shared-memory budget hints relevant to fitting a v5-pixel-temporal-tier model into a single mega-kernel.
- Steam Deck / Vulkan-compute fallback path: explicitly mark "no matrix accelerator on RDNA 2; ceiling is ~25% of the matrix-equipped-vendor peak".
- Open questions list (what we still don't know, what needs a benchmark on hardware we don't have).

Constraints: docs only, no code, no commits to the active scripts. Final commit message suggestion: `docs(notes): vendor optimization audit reference for S6 perf pass`.

## C6 — Custom CUDA mega-kernel design memo

Severity: low (S6 prep)

Deliverable: `docs/superpowers/notes/cuda-mega-kernel-design.md` covering:

- Target shape: single fused kernel for the v5-pixel-temporal forward (encoder backbone → pixel-shuffle upsample → temporal head) at 1080p → 4K on RTX 3080 Ti
- Why fused: kernel-launch overhead per layer dominates at this model size on a small GPU
- Weight-resident-in-shared-memory layout sketch — assume 626K params @ FP16 = ~1.25 MB, fits in 100 KB tiles
- Tensor-core MMA tile shapes (Ampere `mma.sync.m16n8k16` for FP16) for each conv layer
- Open questions (e.g.: where to put the bicubic-skip add; how to vectorize PixelShuffle)
- Reference: list 2–3 academic / NVIDIA blog references for similar fused inference kernels

Constraints: docs only. Final commit message suggestion: `docs(notes): CUDA mega-kernel design memo for S6 fused inference`.

## C7 — README S5 status update

Severity: medium (visible to anyone reading the repo)

`README.md` currently says S5 is "design committed, impl pending" or similar. Update the sprint progression table row to reflect the live state:

- Implementation phase complete (both tracks, 87 tests green pre-Codex round-3 patches → 90+ green now after C3/C4 fixes)
- Pixel training run launched 2026-05-04 17:20 CDT (PID 2360 on `<train-host>`)
- TartanAir-only because Sintel Depth subset is not yet downloaded (separate fetch — see C5/runbook §3)
- Gaussian training run pending pixel completion per the sequential-GPU directive
- Closeout (Plan Task 10 / Task 14) blocks on the held-out eval after training finishes (~04:30 CDT 2026-05-05)

Constraints: do NOT prematurely claim a v5 ship. The status is "training in flight, results pending."

Final commit message suggestion: `docs(readme): S5 status — pixel training in flight, Gaussian queued`.

## C8 — Optional: pixel Phase-2 transition checkpoint visualization

Severity: low (nice-to-have; only if there's idle time)

When the pixel training crosses step 10000 (Phase 1 → Phase 2 transition; backbone unfreezes + temporal-consistency loss activates), it'd be useful to render the model output on a fixed test frame both before (`step-00010000.pt`) and after (`step-00012000.pt`) the transition to confirm the unfreezing actually moves the weights. A side-by-side PNG would be a 30-line script. Skip if Sintel Depth, vendor notes, or CUDA memo are higher value.

Deliverable: `scripts/sr_temporal_phase_transition_diff.py` taking two ckpt paths + an LR test frame and writing `before.png`, `after.png`, `diff.png` (absolute difference scaled).

Final commit message suggestion: `v5-pixel(sr): phase-transition diff visualizer for sanity-checking unfreeze`.
