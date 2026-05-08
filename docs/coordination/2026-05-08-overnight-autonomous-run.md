# 2026-05-08 overnight autonomous run plan

**Operator:** going to sleep, gave authorization to multi-thread aggressively.

**Mandate (verbatim):**
- multi thread as much as possible
- loop and improve
- if NVIDIA custom kernel finishes → move to AMD, then Intel
- don't stop
- use subagents and codex (codex using subagents)
- if questions arise: build the most you can without exact answers, keep going
- if you want another track: work on the capture tool
- bonus: Playwright control my entire 3080ti pc and autonomously test
- authorized to do what's needed within reason

## Active tracks (start of night)

| Track | State | Next |
|---|---|---|
| **Phase 3a — rasterizer dfeat backward** | 🔄 codex running | wait → dispatch 3b on commit |
| v6.1 trainer (PID 22712) | ✅ running with retuned λ | leave alone, monitor for divergence only |
| Held-out trajectory eval (PID 29232) | 🔄 grinding ckpt-1000 | leave alone, results land async |
| Lambda retune signals | ✅ trending right at +360 steps | watch at step ~9000 |

## Pipeline (sequential — same files)

`Phase 3a → Phase 3b → Phase 3c → Phase 3d → "kernels green" → AMD HIP port plan → AMD HIP port phases → Intel SYCL port plan → Intel SYCL port phases`

## Parallel tracks (different files, can run concurrent)

| Track | Files | Spawn order |
|---|---|---|
| Capture tool advancement | `server/oss_capture_ingest/`, `scripts/build_capture_installer.py`, `oss_capture/` (DLL — not yet) | parallel with Phase 3 |
| Cross-engine eval script | `scripts/sr_cross_engine_eval.py` (NEW) | parallel |
| Playwright 3080ti automation | `scripts/3080ti/` + `tests/playwright/` (NEW) | parallel |
| Dashboard backlog (B11/B12/B26/B27) | `scripts/build_public_dashboard.py` + new files | parallel WHEN Phase 3 frees |
| AMD HIP port | `oss/hip/` (NEW) | after Phase 3 done |
| Intel SYCL port | `oss/sycl/` (NEW) | after Phase 3 done |

## Decision rules during night (operator absent)

1. **If a question requires operator input:** make the conservative choice + document the assumption in the commit body. Don't block.
2. **If a codex build fails:** retry once; if still failing, dispatch a `fix-` codex with the failure log. Don't block on the broken commit.
3. **If a sanitizer/test gate fails:** revert the commit + dispatch a fix-codex. Don't push broken work to origin/main.
4. **If trainer (PID 22712 on 3080 Ti) crashes:** detect via metrics.json mtime > 5 min. Restart via the same WMI orphan-spawn pattern (`/Users/cashconway/Library/.../launch-trainer.ps1`). Auto-resumes from latest ckpt.
5. **Document EVERY destructive action** in this file's "actions log" section before doing it.
6. **At end of night:** write a status memo at `docs/coordination/2026-05-08-overnight-results.md` so morning-operator can catch up in 5 minutes.

## Boundaries (NOT authorized)

- Don't touch the trainer's checkpoints destructively.
- Don't push broken commits to origin/main.
- Don't disturb the in-flight v6.1-pico-001 lambda recovery — flip kernel default ONLY for next clean training run (per Phase 3 plan J.5).
- Don't dispatch `/ultrareview` or other billing-heavy operations.
- Don't reach beyond cashcon57's repos / tailnet hosts / R2 buckets.

## Wakeup cadence

- 5-min interval during active dispatches
- 10-min interval during long codex runs (>30min remaining)
- Don't sleep past 30 min — operator may wake unexpectedly

## Actions log (chronological)

(filled in as the night proceeds)

## Operator add-on (sent before sleep)

> "Make a way on the dashboard to checkbox options in the viz and see them blown up side by side, much larger, instead of being a small part of the entire film strip. make that intuitive."

Tracked as new dashboard task: B-viz-blowup. Acceptance: visitor checks 2-N source columns in the viz strip (LR / bicubic / v5 / v6.1 / GT / err-v5 / err-v6.1) → those columns render LARGE, side-by-side, in a dedicated viewer panel. Single column = full viewport width. Two columns = side-by-side with synchronized pan/zoom. Three+ columns = grid. Intuitive UX: checkboxes overlaid on each viz column with a "Compare selected" CTA.

Dispatching as parallel codex now (touches index.html only, disjoint from Phase 3a's oss/cuda/ files).

## Autonomous queue (16 prompts loaded at 02:02 local)

Queue runner: `scripts/codex_queue_runner.sh` (PID 90522, nohup'd). Logs to `/tmp/codex-queue.log`. Dispatches sequentially; survives macOS sleep cycles.

Order:
1. `010_cuda-phase3b-dxy-dconic` — extends rasterize_backward kernel
2. `020_cuda-phase3c-postpass` — adds conic→(scale,rot) post-pass + drops NotImplementedError
3. `030_cuda-phase3d-paritytrain` — 1k-step parity training acceptance
4. `040_dashboard-b21-glossary` — glossary popovers (C6)
5. `050_dashboard-b22-since-last-visit` — what's-new banner (C7)
6. `060_dashboard-b27-status-page` — 5-service health page (X5)
7. `070_dashboard-b23-embed-cards` — embed.html + embed buttons (C11)
8. `080_capture-c1-simulator` — operator test harness
9. `090_capture-c5-uploader-edge-cases` — 4xx/5xx/timeout coverage + 429 fix
10. `100_capture-c6-build-index` — daily R2 _index.parquet roll-up
11. `110_capture-c4-readmes` — install README + supported-games policy
12. `120_playwright-p1-mvp-smoke` — Playwright 3080 Ti MVP smoke
13. `130_dashboard-b11-cost-panel` — cost panel + Sponsor button + cloud-GPU projection
14. `140_dashboard-b12-repro-manifest` — reproducibility manifest per ckpt
15. `150_dashboard-b14-gpu-mem-chart` — GPU memory rolling 30-min chart
16. `160_dashboard-b18-divergence-alarm` — loss-divergence alarm

Stop the queue (graceful): `touch /tmp/codex-queue/.stop`
Kill: `pkill -f codex_queue_runner.sh`
Tail: `tail -f /tmp/codex-queue.log`

