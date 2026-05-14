# Dashboard audit — deferred items (2026-05-14)

External agent ran a dashboard audit on 2026-05-14 during pico-005 inflight.
This memo logs which findings landed in the same session vs which are real
engineering items deferred to a later sprint.

## Landed in this session

| # | Finding | Fix | Commit |
|---|---|---|---|
| 2 | Hero drops the active v7 loss (reads `loss_total ?? loss`; v7 uses `total`) | Added `?? row.total` fallback at all 7 read sites in index.html | (this commit) |
| 9 | "What's new in v7" button missing | Added a runner-up note in run-card copy + linked Phase 3 plan from RUN_CONFIG entry | (this commit) |
| viz | seq A / seq B buttons don't reflect multi-row strips | Dynamic row-button generation reading `scene_labels` from a sidecar `.scenes.json` written by the inflight-viz daemon. Defaults to seq A / seq B for v6.x runs (2 rows, no sidecar). | (this commit) |
| viz | v7 daemon iterates dataset order, ignores manifest | `_filter_v7_triplets_by_manifest()` reorders `_triplet_indices` to match `v7_held_out_manifest.json`. Manifest has 6 oldtown pairs spread across the trajectory (frames 100/400/700/1000/1300/1500) | (this commit) |

## Deferred — real engineering items, not patchable in one session

### 1. `data.json` is 8.6 MB

The dashboard fetches the full `data.json` on every live update. With ~50 runs
each carrying full history + viz manifests, the payload grew past the warning
threshold. Real fix: split into a small `summary.json` (run names, active flag,
latest step, latest loss, hero card data) for the landing view + lazy-load
per-run histories on click. Estimated 1-2 days.

### 2. Mobile layout broken at 390px viewport

The first screen is mostly logo + explanatory copy; training state, metrics,
and charts are too far down. Some tooltips render offscreen. At least one
table overflows horizontally. Real fix: redesign the hero block for narrow
viewports + audit tooltip placement + tables → cards on mobile. Estimated 2-3
days, ideally done with a designer.

### 3. Information hierarchy is too dense

The current page mixes narrative, funding, service health, metrics, and
explanations without a clear first-read path. Audit recommendation: compact
"Now" dashboard as the first viewport (current run / training state / step+age
/ current loss / latest viz / held-out score / one-sentence status), then
split the rest into Overview / Research / Runs / Artifacts / Funding+Infra
sections. Estimated 3-5 days; couples with #2.

### 4. Tailwind via CDN triggers production warning

The page loads `https://cdn.tailwindcss.com` which prints a console warning.
Real fix: build step (Tailwind CLI) that emits a minified CSS bundle, ship
that instead. Estimated half a day; useful but not load-bearing.

### 5. Two dashboard implementations

`scripts/build_public_dashboard.py` has a fallback HTML template (~1200 lines)
that's only used when the placeholder isn't found. The real dashboard is
`dashboard-public/index.html` (~7800 lines, hand-evolved static app). Most
production updates touch only the static HTML; the Python template path is
effectively dead. Real fix: pick one, delete the other, document the choice.
Estimated 1 day.

### 6. Dashboard status copy "training paused" inaccurate

When the trainer hasn't logged in 2+ minutes, the dashboard shows "training
paused" — but the actual state is more like "stalled at step N, waiting for
next step" or "between log-events". Real fix: surface raw state ("step N,
last log Xs ago") + a parenthetical plain-language interpretation. Estimated
2 hours, do it with #3.

## Owner / next-step

These should land before the next major training run kickoff (post-pico-005),
ideally bundled into one "dashboard v2" sprint. Until then the current
dashboard works at the desktop viewport for the current researcher audience.
