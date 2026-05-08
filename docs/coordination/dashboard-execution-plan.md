# Dashboard Execution Plan

Source backlog: `/Users/cashconway/OpenSuperSampling/docs/coordination/dashboard-roadmap.md` (triaged 2026-05-07).
Hard rule: every dashboard change must pass Playwright visual + programmatic schema/value checks BEFORE push to `origin/main` → R2. Source: `~/.claude/projects/-Users-cashconway-OpenSuperSampling/memory/feedback_dashboard_visual_test.md`.

This plan covers only YES (✅) and CONDITIONAL (🟡) items. DISCUSS (💬) items are listed at the end as escalations.

## YES/CONDITIONAL inventory

YES (24): O1, O3, O4, O7, O10, O11, O12, C2, C4, C5, C6, C7, C11, C12, R1, R2, R3, R4, R5, R6, R7, R8, R10, R11, R12, X2, X3, X5, X7
CONDITIONAL (6): O2, O8, O9, C3, C9, R9, X1
(O7, R4, R10, X3, C8 carry NEW sub-asks — C8 is DISCUSS so excluded.)

## File-conflict map (the merge bottleneck)

`dashboard-public/index.html` is touched by virtually every visual item. To avoid stomping commits, items that edit `index.html` are sequenced inside one batch (codex slot only one at a time). Disjoint files (build_public_dashboard.py-only, new model_card.yaml, new architecture.svg, RSS feed file) can run in parallel against an in-flight index.html batch.

| File | Items |
|---|---|
| `dashboard-public/index.html` | O1, O3, O4, O7, O10, O12, C2, C4, C5, C6, C7, C11, C12, R1, R2, R3, R7, R8, R9, R10, R11, R12, X1, X3, X5, X7 |
| `scripts/build_public_dashboard.py` | O11, R4, R5, R6, X2, X7, plus enrichment for R1/R2/R3/R7 |
| New static asset files | C3 (rss.xml emitter), C2 (architecture.json + svg), R5 (model_card.yaml), X7 (dataset_card.yaml) |
| New scripts | O2 (memo publisher), O11 (manifest emitter), R10 (FFT precompute) |

## Foundation work (must land first)

These three blockers unlock everything else and are sequenced at the front:

| Order | Slug | Owner | Scope |
|---|---|---|---|
| F1 | `dash-x2-schema-versioning` | Codex | Add `schema_version: "2026-05-07"` to data.json; add `tools/check_data_schema.py` that validates required keys (runs[], score_log shape, etc.); CI hook. |
| F2 | `dash-r12-url-state` | Codex | URL query state (`?run=&step=&chart=&zoom=`) + parser/serializer in index.html. Foundation for citeable views. |
| F3 | `dash-score-log-enrichment` | Codex | Extend `build_public_dashboard.py` `slim_row()` to emit per-frame deltas, std/IQR, and a `models[]` field with PSNR/LPIPS coords (for R1 Pareto). Update schema check. |

After F1–F3 ship, batches B1..Bn run with maximum parallelism.

## Batch table (post-foundation)

Each row is one codex dispatch (or Claude/sub-agent task). Effort target: 30–60 min, single coherent commit. Dependencies referenced by batch ID.

| Batch | Slug | Owner | Items | Files (primary) | Depends on | Est. min |
|---|---|---|---|---|---|---|
| F1 | dash-x2-schema-versioning | Codex | X2 | build_public_dashboard.py, tools/check_data_schema.py | — | 35 |
| F2 | dash-r12-url-state | Codex | R12 | index.html | F1 | 45 |
| F3 | dash-score-log-enrichment | Codex | (enabler) | build_public_dashboard.py, schema check | F1 | 50 |
| B1 | dash-r5-model-card | Codex | R5 | model_card.yaml, build_public_dashboard.py, index.html (model card panel) | F1 | 50 |
| B2 | dash-x7-dataset-card | Codex | X7 | dataset_card.yaml, build_public_dashboard.py, index.html | F1, B1 (UI panel layout) | 45 |
| B3 | dash-c2-architecture | Codex | C2 | architecture.json, architecture.svg, index.html | F2 (deep-link block) | 60 |
| B4 | dash-r1-pareto | Codex | R1 | index.html (new chart) | F2, F3 | 55 |
| B5 | dash-r2-r3-perframe-ci | Codex | R2, R3 | index.html | F3, B4 (chart infra reuse) | 60 |
| B6 | dash-r7-failure-mode | Codex | R7 | index.html, build_public_dashboard.py (failure_frames[] field) | F3 | 50 |
| B7 | dash-r8-loss-decomp-toggle | Codex | R8 | index.html | F2 | 30 |
| B8 | dash-r11-heatmap | Codex | R11 | index.html, build_public_dashboard.py (heatmap_url field) | F3 | 55 |
| B9 | dash-r10-fft | Codex | R10 | scripts/sr_fft_precompute.py, build_public_dashboard.py, index.html | F3 | 60 |
| B10 | dash-r6-ood | Codex | R6 | scripts/ood_score.py, build_public_dashboard.py, index.html | F3 | 55 |
| B11 | dash-r4-cost-panel | Codex | R4 + 2 NEW (Sponsor btn, cloud projection) | build_public_dashboard.py, index.html | F1 | 55 |
| B12 | dash-o11-repro-manifest | Codex | O11 | scripts/emit_repro_manifest.py, build_public_dashboard.py, index.html (manifest drawer) | F1 | 50 |
| B13 | dash-o10-abort-reason | Codex | O10 | build_public_dashboard.py, index.html | F1 | 30 |
| B14 | dash-o12-gpu-mem-chart | Codex | O12 | build_public_dashboard.py (gpu_mem rolling), index.html | F1 | 45 |
| B15 | dash-o3-train-log-tail | Codex | O3 | build_public_dashboard.py (tail), index.html drawer | F1 | 40 |
| B16 | dash-o7-eta-extrapolation | Codex | O7 + NEW future trend | index.html | F2, F3 | 55 |
| B17 | dash-o4-ckpt-diff | Codex | O4 | index.html | F2 (deep-linkable diff URL) | 60 |
| B18 | dash-o1-divergence-alarm | Codex | O1 | index.html | F3 | 35 |
| B19 | dash-c4-progress-bar | Codex | C4 | index.html | F1 | 25 |
| B20 | dash-c5-citation | Codex | C5 | index.html | — | 20 |
| B21 | dash-c6-glossary | Codex | C6 | index.html | — | 30 |
| B22 | dash-c7-since-last-visit | Codex | C7 | index.html | F1 | 35 |
| B23 | dash-c11-embed-cards | Codex | C11 | index.html (+ embed.html) | F2 | 50 |
| B24 | dash-c12-frame-viewer-deeplink | Codex | C12 | index.html | F2 | 35 |
| B25 | dash-x1-dataset-preview | Codex (CONDITIONAL) | X1 | index.html, R2 upload of ~32 MB frames | B2 | 40 |
| B26 | dash-x3-multirun-help | Codex | X3 + NEW Help button | build_public_dashboard.py (multi-run aggregation), index.html | F1 | 55 |
| B27 | dash-x5-status-page | Codex | X5 | scripts/status_probe.py, status.json, index.html | F1 | 50 |
| B28 | dash-r9-ab-slider | Codex (CONDITIONAL) | R9 | index.html | F2 | 50 |
| B29 | dash-o2-step-memos | Codex (CONDITIONAL) | O2 | scripts/memo_publish.py, docs/lab/, build_public_dashboard.py, index.html | F2; needs publish-vs-draft gate (Claude-main spec) | 60 |
| B30 | dash-o8-train-control | Claude-subagent then Codex (CONDITIONAL) | O8 | Worker (oss_dashboard_uploader_worker.js) tailnet IP gate; index.html control hidden by default | Security spec from Claude-subagent | 60 |
| B31 | dash-o9-hyperparam-diff | Codex (CONDITIONAL) | O9 | build_public_dashboard.py (whitelist filter), index.html | F1 + whitelist authored by Claude-subagent | 50 |
| B32 | dash-c3-rss-feed | Codex (CONDITIONAL) | C3 | scripts/emit_milestones_rss.py, milestones.xml, index.html link | F1 | 35 |
| B33 | dash-c9-community-link | Claude-main decision then Codex | C9 | index.html | Operator picks Matrix vs GH Discussions | 15 |

Claude-subagent jobs (research/spec, no code commits):
- S1: Author O8 tailnet IP-gate threat model + whitelist of safe operator-control endpoints. Output: `docs/coordination/o8-control-spec.md`.
- S2: Author O9 hyperparam-publish whitelist (env-var/secret stripping rules). Output: `docs/coordination/o9-hyperparam-whitelist.md`.
- S3: Review codex output of B4 (R1 Pareto) and B5 (R2/R3) for ML-figure correctness (axes log/linear, Wilson CI formula, IQR symbol). One sub-agent per review pass.
- S4: Draft each codex prompt file from this plan into `/tmp/prompt-dashboard-<slug>.txt` before dispatch.

Claude-main responsibilities:
- Final architectural decisions (C9 community channel pick, X1 dataset frame selection).
- Final `git push origin main` after Playwright + programmatic checks pass on each batch.
- User-facing Q&A for the DISCUSS list before any of those move into batches.

## Verification contract (applies to every batch)

Each codex prompt MUST embed a verification block. Pass/fail blocks merge.

### A. Programmatic schema check (always)
```
python tools/check_data_schema.py dashboard-public/data.json
# exits non-zero if schema_version missing, required keys absent, or
# field types diverge from the contract for the batch's new fields.
```

### B. Playwright visual capture (always)
Capture two viewports per change:
```
npx playwright test --grep "<slug>" \
  --reporter=list \
  -- --viewport=1440x900 --viewport=390x844
# screenshots stored at artifacts/<slug>-desktop.png + -mobile.png
```

### C. Per-batch programmatic assertions

| Batch | Programmatic assertion (in addition to A+B) |
|---|---|
| F1 | `jq -e '.schema_version=="2026-05-07"' data.json`; tools/check_data_schema.py exits 0 |
| F2 | URL `?run=v6.1&step=5000&chart=loss-decomp&zoom=1500-3000` loads; `window.__getDashboardState()` returns matching state; back/forward restores |
| F3 | `jq -e '.runs[0].score_log[0]\|has("psnr") and has("lpips") and has("frame_idx")'` |
| B1 | Model card panel renders fields from `model_card.yaml`; missing-field check fails build |
| B3 | Click each block in architecture.svg → drawer opens with non-empty memo + repo link; broken-link assert via fetch HEAD |
| B4 | Pareto chart has ≥2 model points; axis labels = "PSNR (dB)" and "LPIPS"; trajectory path length > 0 |
| B5 | Per-frame histogram has N bins matching frame_count; Wilson CI formula spot-check vs known-good fixture |
| B6 | Failure-frame grid renders K worst frames; each links to deep-link (uses F2) |
| B7 | Toggle between absolute/fraction-of-total reflows chart; sum-of-fractions ≈ 1.0 |
| B8 | Heatmap PNG referenced from data.json HEAD-fetches 200 |
| B9 | FFT panel: spectrum array length == precomputed bins |
| B11 | Cost panel renders kWh/USD/GPU-h; cloud-projection table has 4 GPU rows; Sponsor button links to repo SPONSORS file |
| B12 | Manifest drawer shows git_sha matching latest commit; "recreate" command copy-button copies to clipboard |
| B14 | gpu_mem chart renders ≥30 points |
| B16 | Dashed extrapolation line extends past last-data-point; confidence cone present |
| B18 | Synthetic 3σ loss spike → tab title flips to "⚠ DIVERGED" |
| B19 | Progress bar % == steps/total_steps |
| B22 | Fresh localStorage → "What's new" badge shows; second visit hides it |
| B23 | Embed iframe URL renders just the chart (no nav) |
| B26 | Multi-run aggregator returns >1 active run when seeded; Help button anchors to contribute section |
| B27 | status.json drives 5 service rows; healthy/red icon flips on synthetic outage |

### D. Push gate
A batch can push to `origin/main` only when:
1. `tools/check_data_schema.py` exits 0 against the freshly built `data.json`.
2. Both Playwright screenshots saved with no console errors (`page.on('console')` asserts no `error` events).
3. Per-batch assertion (table C) passes.
4. Diff of `dashboard-public/index.html` and `scripts/build_public_dashboard.py` reviewed by Claude-main or a Claude-subagent (S3).

If any check fails, the codex session retries (max 2) before kicking back to Claude-main for triage.

## Schema-drift hazard map (the multi-week-outage class of bug)

Every batch that adds a new data.json field gets a contract test wired into `tools/check_data_schema.py` in the SAME commit. Items that introduce new fields:

| Batch | New data.json field(s) | Contract assertion added in same PR |
|---|---|---|
| F1 | `schema_version` | required string |
| F3 | `runs[].score_log[].{psnr,lpips,frame_idx,delta_vs_bicubic}` | required numbers |
| B1 | `runs[].model_card` | required object with whitelisted keys |
| B2 | `dataset_card` (top level) | required object |
| B6 | `runs[].failure_frames[]` | array of `{frame_idx,delta,thumb_url}` |
| B8 | `runs[].heatmap_url` | URL string, HEAD 200 |
| B9 | `runs[].fft.{bins,oss,bicubic,...}` | array length match |
| B10 | `runs[].ood[]` | array of `{scene,score}` |
| B11 | `runs[].cost.{kwh,usd,gpu_hours,projections}` | required numbers |
| B12 | `runs[].repro_manifest` | required object |
| B14 | `runs[].gpu_mem_log[]` | array of `[t,mb]` |
| B27 | `status` (top level) | object of 5 service keys |

Rule (codify in every prompt): "If you add a field to data.json, you MUST add an assertion for it to `tools/check_data_schema.py` in the same commit. PRs that add fields without an assertion will be reverted."

This is the single intervention that prevents recurrence of the multi-week silent breakage.

## Dependencies / sequencing summary

```
F1 ─┬─► F2 ─┬─► B3, B7, B17, B23, B24, B28, B29, R12-dependent UI
    │       │
    │       └─► B16, B5 (also need F3)
    │
    └─► F3 ─► B4 ─► B5
         │
         ├─► B6, B8, B9, B10, B14, B16, B18
         │
         └─► (most chart batches)

B1 ─► B2 ─► B25
S1 ─► B30
S2 ─► B31
Operator decision ─► B33, B25 (frame pick)
```

## Capacity model + ETA

The bottleneck is `dashboard-public/index.html`. Concurrent codex sessions editing it produce conflicting commits — sequence them.

Parallelism rules:
- At any moment, ONE codex session may be editing index.html. (`-index-lock`.)
- Concurrently, up to TWO additional codex sessions may run if they edit only disjoint files (build_public_dashboard.py-only, or new asset files like architecture.json / model_card.yaml / scripts/*).
- Claude-subagents (S1, S2, S3, S4) run in parallel without contention since they only write coordination docs and prompt files.

Concurrency cap: **1 index.html-touching session + 2 disjoint sessions = 3 codex sessions concurrent**, plus N Claude subagents.

Wallclock estimate (avg batch 45 min, 33 batches incl. F1–F3):

- **Sequential**: 33 × 45 = ~24.75 hours of codex wallclock.
- **Max-parallel** (1 index + 2 disjoint, ignoring foundation serialization): index-bound critical path ≈ 26 index-touching batches × 45 min = 19.5 hr; with 2 disjoint lanes covering the 7 non-index batches in ~2 parallel hr; total ≈ **20 hours** wallclock for codex, dominated by the index.html lane.
- **Foundation-aware**: F1, F3 land first sequentially (~85 min) → after that the parallel pipeline starts. Total ≈ **21 hr** wallclock max-parallel.

Practical day-shape: ~3 codex hr × 7 working days, plus ~30 min/day Claude-main review = ~7 calendar days from kickoff, assuming clean reviews.

## DISCUSS items (escalate to operator BEFORE more dispatches)

These are blocked on operator Q&A and are NOT in any batch:

1. **O5** Ckpt download buttons — storage cost in R2 + which ckpts to publish (auto-all vs operator-blessed-only?).
2. **O6** Cost meter — ship at all? Only paired with R4 cost projection?
3. **C1** Try-it demo widget — self-host inference vs client-side WebGPU? Cost ceiling?
4. **C8** Latency/throughput per GPU class — write custom NVIDIA kernel FIRST? Multi-week detour confirm/decline.
5. **C10** Public roadmap content (Now/Next/Later/Wishlist) — needs interactive Q&A on actual content.
6. **X4** Public API — auth shape (Bearer vs anonymous-rate-limited)? Throttle ceiling?
7. **X6** Audit log — public visibility (count-only vs full)?

Once any of these are resolved, add a new batch row to the table above and dispatch.

## Known constraints to keep top-of-mind

- "Quality is paramount." — don't skip the verification gate to chase wallclock.
- Codex carries the bulk to save Claude tokens; Claude-main only for orchestration + final push approval.
- Schema-drift is the historic-breakage failure mode: every new field gets a contract test in the same commit.
- The 3080ti repo at `E:\oss-gaussian-server` is NOT touched by these batches (dashboard work is Mac-only); ci_auto_heal sync continues independently.
