# Dashboard layout audit — empirical verification

**Date:** 2026-05-14
**Author:** Claude (verification pass)
**Source audit:** handoff from prior session (anonymous agent)
**Live URL audited:** https://opensupersampling.com/
**Local source audited:** /tmp/v7_main9 worktree on `v7-chart-keys-fix` (tracks origin/main, HEAD `1d2181c`)

This memo records empirical verification of the seven audit claims. Every
claim was checked against the live production site via Playwright (mobile
390x844 and desktop 1440x900) and against the current source at
`dashboard-public/index.html`. No fix is landed by this memo; the next
section proposes a consolidated layout change for user review.

## Verification matrix

| # | Audit claim | Verdict | Evidence |
|---|---|---|---|
| 1 | Viz strip duplication (top-level + per-run) | TRUE | DOM contains nine "Viz strips" headings: one `<h2>` in `#global-viz-section` (top 5200px mobile) plus eight `<h3>` inside per-run lineage accordions. Labels and controls overlap, not just placement. |
| 2 | Top-level viz strip is too late | TRUE | Mobile (390px): `#global-viz-section` top = 5179px, after hero (1295), at-a-glance (1179), teacher-student (3945). Desktop (1440px): top = 2637px. |
| 3 | Global viz controls overflow on mobile | TRUE | Viewport 390px. `<select>` runs to right=405. "Side-by-side" button right=473. "Play" right=535. fps slider right=768. fps label right=780. Audit's "~780" was precise. Twelve descendants of `#global-viz-section` overflow viewport. |
| 4 | Active v7 has very few strips, default state underwhelming | TRUE (refined) | Live `data.json`: `srcnn-v7.0-pico-005` has 2 viz strips at step 500 (audit said 1; now 2). `v6.2` has 149. The "Play" / strip-gallery affordance defaults to v7 and feels overbuilt. |
| 5 | OSS-FX section visually competes with viz strips | TRUE | `#oss-fx-section` top = 5727px on mobile, immediately after `#global-viz-section` ends at 5660. Two large evidence surfaces stacked; OSS-FX shows v6.2 step 70k while page headline is v7. |
| 6 | First useful read starts too low on mobile | TRUE | At-a-glance starts at 1179px mobile / 956px desktop. Hero meta is at 1295px mobile (after logo + headline + explanatory paragraphs). |
| 7 | Production still shows `loss --` for v7 | TRUE | Hero `#hero-active-meta` renders verbatim: `step 500 / loss -- ↓ lower is better`. Live data has `latest_metrics.total = 0.3094` (no `loss_total` / `loss` keys for v7). |

### Audit was wrong about one detail

The audit asserted the local file was already fixed at line 7003. That is
not the case in the current `origin/main` head:

- `dashboard-public/index.html:7247-7248` still reads
  `const latest = active.latest_metrics || {}; ... fmtNumber(latest.loss_total ?? latest.loss, 5)`.
  No `?? latest.total` fallback. v7 only emits `total` (and `sr_charbonnier` / `sr_lpips`).
- `dashboard-public/index.html:3642-3643` (`headlineValue`, `value_from === "loss_total"`)
  has the same bug — same two-key fallback chain.

So the `loss --` regression is not a stale-deploy problem; it is a missing
v7 fallback in the current source. A fix must add `?? latest.total` (or
`?? latest.sr_charbonnier`) to both call sites.

## Trainer health check (sanity)

- `srcnn-v7.0-pico-005` alive on `3080ti-windows`, history.jsonl appending.
- Last three rows: step 400 total=0.2563, step 450 total=0.1031, step 500 total=0.3094 (bumpy but expected during warmup; lambda_fg/lambda_temp still 0).
- VRAM 2991 / 12288 MiB (~24%) at step 500. Target ~60% post-warmup is gated on canvas expansion: `canvas_count = 1024 / capacity 16384`. Re-check after step 5000.

## Proposed consolidated layout change

Goal: collapse the viz experience into one clear model, surface the
first useful read above the fold, and silence mobile control overflow.

### F1. Hero / first-screen rewrite (mobile-first)

Above 850px on mobile, render only:

1. One-line status: `v7 is training · step 500 / 100,000 · loss 0.30939 · viz @ step 100`
2. "What's new in v7?" disclosure button (collapses architecture intro behind it; intro is currently above the fold).
3. Latest-visual thumbnail (single most recent viz frame for the active run; tap to expand into visual-progress section).

Wire the hero meta to `latest_metrics.loss_total ?? latest_metrics.loss ?? latest_metrics.total` so v7 stops rendering `--`. Same fix in `headlineValue`.

### F2. Single viz surface: rename + dedupe

- Rename `#global-viz-section` → `Visual Progress`. It becomes the only general viz browser.
- In each run accordion, replace the full `Viz strips` card with a single button: `Open visual history for this run` that scrolls/focuses the Visual Progress section to that run (or opens a drawer).
- Removes 8 of 9 viz-strip surfaces from the page. Lineage stays a navigation surface, not a viz surface.

### F3. Mobile controls reflow

Inside `#global-viz-section`, stack controls vertically on `max-width: 480px`:

```
[ run selector, full width ]
[ Compare ][ ▶ Play ]
[ fps slider — only after Play pressed ]
```

Hide "Play" entirely when the selected run has < 2 strips (matches v7 reality today).

### F4. OSS-FX vs viz separation

OSS-FX is currently the second large evidence surface immediately below
global-viz. Either:

- (a) Move OSS-FX below the hypothesis / model-card sections so the two
  evidence surfaces are not adjacent, OR
- (b) Make OSS-FX a tab inside `Visual Progress` (run-selector option
  `v7.0 Pico OSS-FX`) so both share one player.

Recommend (b): one player, one set of controls, runs as dropdown options.

### F5. data.json contract patch

Add `tools/check_data_schema.py` assertion that every `runs[*].latest_metrics`
exposes one of `loss_total`, `loss`, or `total`. This is the F1 schema-drift
contract from the dashboard execution plan.

## Verification gate before push

Per the project's hard rule
(`feedback_dashboard_visual_test.md`): every change above requires
Playwright screenshot at 390x844 + 1440x900 AND a programmatic data-shape
check before pushing `v7-chart-keys-fix:main`. The `loss --` fix is the
single highest-priority item — it is a one-line edit at 7248 + 3643 and
should ship in its own commit ahead of the layout work.

## Open questions for the user

1. Drawer vs in-place focus for "Open visual history for this run"? Drawer
   is faster to ship; in-place focus is more discoverable.
2. Keep "What's new in v7?" as a disclosure, or as a separate landing
   `/v7` route? Disclosure preserves single-page bookmarkability.
3. Move OSS-FX into Visual Progress as a run option (F4b) or leave as its
   own section but push it past the hypothesis surface (F4a)?

These three are the only DISCUSS items; everything else above is a
verified fact + a derivable proposal.
