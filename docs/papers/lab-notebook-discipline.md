# Lab Notebook Discipline

Rules for keeping the experiment record paper-ready. Followed by the human, by every Claude / Codex / GPT agent, and by every CI artifact.

## The single rule

**Every training run, every benchmark, every architectural ablation gets a memo at `docs/superpowers/experiments/YYYY-MM-DD-<slug>.md` BEFORE the result is allowed to influence a downstream decision.**

If a number isn't in a memo, it doesn't exist for the paper.

## Memo template

Save as `docs/superpowers/experiments/YYYY-MM-DD-<short-slug>.md`.

```markdown
# <Topic>
**Date:** YYYY-MM-DD
**Status:** [in-progress | complete | superseded by <memo-path>]
**Predecessor:** [path to prior memo, if any]
**Hardware:** [3080 Ti / M3 Max / Lambda H100 / etc.]
**Code commit:** [git SHA at start of run]

## Hypothesis

One sentence. What did we expect to see, and why?

## Setup

- **Data:** dataset name, scene/sequence filter, train/val split.
- **Hyperparams:** tier, batch size, learning rate, optimiser, schedule, gradient clip, seed.
- **LR-synth config:** σ, jitter on/off, jpeg on/off + quality.
- **Loss:** wired terms (L1, SSIM, LPIPS, …) + weights.
- **CLI used (verbatim):**
  ```
  python -m oss.gaussian.train.train --tier ... --dataset ... ...
  ```

## Result

| Step | Loss | model_PSNR | bicubic_PSNR | Notes |
|------|------|-----------:|-------------:|-------|
| 1000 | 0.20 | 13.6 | 28.2 | first eval |
| ...  | ...  | ...  | ...  |       |

(Optional) Figures committed to `docs/papers/figures/<slug>-<chart>.png`.

## Decision

What changes downstream:
- [ ] Hyperparam X is set to Y for future runs.
- [ ] Architecture component Z is dropped / kept / re-tested.
- [ ] Next experiment is `<path-to-followup-memo>`.

## Open questions

- ...
```

## What counts as "a run that needs a memo"

- Every training run that completes ≥100 steps.
- Every architectural change that ships behind a flag (e.g., `--enable-gbuffer-bias`).
- Every dataset addition or loader change that affects a numeric result.
- Every gradient probe / data probe / sanity script.

## What does NOT need a memo

- Bug fixes that don't change a measured number.
- Doc edits.
- Refactors.
- Aborted runs killed before any eval (note in commit message; don't write a memo).

## Behavioural rules for agents

(These also apply to the human, but agents drift faster.)

1. **Cite memo paths in commit messages.** Example: `addressed in docs/superpowers/experiments/2026-05-02-sprint4-smoke-findings.md §4`.
2. **Update the experiments-index when adding a memo** (`docs/papers/experiments-index.md`). One line per memo, two-tier index by topic and date.
3. **Add a citation to `docs/papers/oss-gaussian.bib` whenever a new paper informs a decision.** Inline-cite from the architecture doc / experiment memo by `\cite{key}`.
4. **When a result contradicts a prior memo, do not delete the prior memo.** Mark it `Status: superseded by <new-memo-path>` and leave the original numbers in place. The negative results matter.
5. **Promote findings to architecture or plan docs eagerly.** A memo finding that doesn't propagate to `gaussian-network-architecture.md` or the relevant Sprint plan within the same agent turn is a memo that will rot.
6. **Never write a paper draft from memory.** Always assemble from memo content.

## When to draft an actual paper

See `docs/papers/README.md` § Drafting cadence. Short version: only when V0 (or later) clears its bicubic-baseline gate. Until then, the memos *are* the paper-in-pieces.
