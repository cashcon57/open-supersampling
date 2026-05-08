# Dashboard Run Allow-List

The public dashboard intentionally shows runs that explain the current model
lineage without flooding the run history with failed smoke tests, data-leak
runs, or pre-v4 prototypes.

## Public Runs

| Run | Status | Reason |
| --- | --- | --- |
| `srcnn-v6.2-pico-002` | Active | Current training run (launched 2026-05-08); v6.2 architecture (DisocclusionSpawner + ConcatFusion + R=16). Default dashboard focus. |
| `srcnn-v6.1-pico-001` | Stopped early | Stippling regression at ~step 14k (2026-05-08); kept for the lineage story. |
| `srcnn-v6-pico-001` | Superseded | Shows the stopped v6 Pico run and the grid-artifact diagnosis that led to v6.1. |
| `srcnn-v6-heavy-001` | Parked | Short HAT-L v6 startup trace; useful context for the heavy branch that was paused before v6.1. |
| `srcnn-v5-pixel-temporal-validated` | Measured | Canonical v5 baseline with held-out PSNR/LPIPS and viz strips. |
| `srcnn-v5-pixel-temporal-clean-restart-override` | Superseded | Earlier v5 attempt with metrics and three viz strips, useful as the pre-validated comparison. |
| `srcnn-prod-v4-lpips` | Baseline | Single-frame v4 baseline and cross-distribution reference point. |

## Excluded Runs

| Run | Reason |
| --- | --- |
| `srcnn-prod-v2` | Too old for the public v4/v5/v6 lineage; no useful mirrored metrics. |
| `srcnn-prod-v3` | Older production baseline superseded by `srcnn-prod-v4-lpips`; omitted to keep the public history focused. |
| `srcnn-v5-gaussian-temporal` | Empty parked directory in the current mirror, with no metrics, scores, or viz to show. |
| `srcnn-v5-pixel-temporal-data-leak-aborted` | Explicitly aborted data-leak run; filtered by the sync deny pattern. |
| `srcnn-v5-pixel-temporal-DOA-distshift-bug` | Known bad distribution-shift bug run; filtered by the sync deny pattern. |
| `srcnn-v5-pixel-temporal-no-lr-synth-aborted` | Aborted run with no informative public artifact. |
| `srcnn-v5-pixel-temporal-preflight-*` | Preflight checks only; not training history. |
