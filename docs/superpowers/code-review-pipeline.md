# OSS-Gaussian Sprint Code Review Pipeline

**Status:** scaffolding complete (v0). Subagent dispatch is the only piece left to wire.
**Scope:** gates the close of each of the 7 OSS-Gaussian sprints.
**Owner:** OSS-Gaussian sprint lead.
**Code location:** [`oss/gaussian/review/`](../../oss/gaussian/review/).

## 1. Why this exists

OSS-Gaussian is a real-time game upscaler being built sprint by sprint, in a
codebase that already ships four pixel-based pipelines (OSS-RG, OSS-SR,
OSS-Pico, OSS-FX). A naive sprint cadence would let regressions and
spec drift accumulate. This pipeline pins down a hard gate: **no sprint
closes without two specialised reviewer agents and a judge agent
agreeing the sprint is safe to merge.**

## 2. Architecture

```
run.py (CLI)
   |
   |-- get_diff(commit_range)      -> diff.patch
   |-- load_sprint_spec(N)
   |
   |-- Reviewer A  (correctness / security / perf / style)
   |       \-> reviewer-a.json
   |
   |-- Reviewer B  (spec / tests / edge / integration / regression)
   |       \-> reviewer-b.json
   |
   \-- Judge       (reads both reports + diff)
           \-> judge.json   { APPROVE | REQUEST_CHANGES | BLOCK }
```

Reviewer A and Reviewer B run with **non-overlapping concerns** so the
two reports can be merged without the judge having to deduplicate.
Each reviewer receives the spec for context, but only one reviewer is
*responsible* for any given dimension.

| Dimension                     | Reviewer A | Reviewer B |
|-------------------------------|:----------:|:----------:|
| Logic correctness             |     X      |            |
| Security (path / injection)   |     X      |            |
| CUDA kernel / Python perf     |     X      |            |
| Idiomatic Python/CUDA/C++     |     X      |            |
| Spec acceptance criteria      |            |     X      |
| Test coverage + meaningfulness|            |     X      |
| Edge cases (NaN, empty, 1-frame) |        |     X      |
| Integration with pixel-based OSS |        |     X      |
| Public API / format regressions |          |     X      |

The judge is **not** a third reviewer. It does no fresh code reading
beyond using the diff to disambiguate reviewer disagreements. Its
output is purely structured triage.

## 3. Output schemas

Reviewer JSON (per finding):

```json
{
  "severity": "high | medium | low",
  "category": "correctness | security | performance | ... ",
  "file": "oss/gaussian/renderer/splat.cu",
  "lines": "L120-L156",
  "issue": "Shared-memory tile read is not coalesced for non-aligned widths.",
  "suggested_fix": "Pad tile width to 32 or use vectorised float4 loads."
}
```

Reviewer report wraps a list of these plus `reviewer_id`, `sprint`, and a
short summary paragraph. Judge output:

```json
{
  "verdict": "APPROVE | REQUEST_CHANGES | BLOCK",
  "rationale": "...",
  "blocking_issues":     [ <Finding>, ... ],
  "non_blocking_issues": [ <Finding>, ... ]
}
```

The judge re-classifies findings into blocking / non-blocking using
its own rubric; severity from the reviewer is advisory.

## 4. Reviewer prompt (A) -- ready to use

```
You are Reviewer A in the OSS-Gaussian sprint review pipeline.

## Your concern: Implementation correctness
Review ONLY: correctness, security (path validation, no injection),
performance (CUDA occupancy, divergence, bandwidth, hot loops),
idiomatic Python / CUDA / C++ style. Do NOT comment on spec
adherence, test coverage, or integration risks -- those belong to
Reviewer B.

## Output
Single JSON object, no prose:
{
  "reviewer_id": "A",
  "sprint": <N>,
  "summary": "<one paragraph>",
  "findings": [
    { "severity": "high|medium|low",
      "category": "correctness|security|performance|style|memory|concurrency",
      "file": "...", "lines": "L<a>-L<b>",
      "issue": "...", "suggested_fix": "..." }
  ]
}

Severity rubric: high = ships a bug / sec hole / >10% perf regression.
medium = correctness smell, missing bounds check, suboptimal kernel.
low = style / naming / minor idiom.
```

## 5. Reviewer prompt (B) -- ready to use

```
You are Reviewer B in the OSS-Gaussian sprint review pipeline.

## Your concern: Spec adherence + integration
Review ONLY: spec acceptance criteria, test coverage and
meaningfulness, edge cases (empty, NaN/Inf, zero-tile, 1-frame,
mixed precision), integration risk against existing pixel-based
OSS-RG / OSS-SR / OSS-Pico / OSS-FX, regression risk on public APIs,
on-disk formats, tensor shapes, CLI flags.

Do NOT comment on low-level correctness, performance, or style.

## Output
Same JSON envelope as Reviewer A but with "reviewer_id": "B" and
category drawn from: spec_adherence | test_coverage | edge_case |
integration_risk | regression_risk.

Severity rubric: high = spec criterion missing OR breaks existing
pipeline. medium = weak coverage on a new path or plausible edge case
unaddressed. low = naming consistency / doc gap.
```

## 6. Judge prompt -- ready to use

```
You are the Judge. Two reviewers (A: correctness; B: spec+integration)
have submitted JSON findings. You do NOT re-review the code; you weigh
the findings against the diff and decide whether sprint <N> closes.

Decision rubric:
- APPROVE         -- no highs, mediums all non-blocking style/test gaps.
- REQUEST_CHANGES -- some medium/high findings fixable in <= 1 day.
- BLOCK           -- a high requiring redesign, OR reviewers disagree
                     materially and you cannot resolve from the diff.

Conflict rule: integration concerns beat performance optimism;
correctness bugs beat "spec satisfied".

Output single JSON object:
{ "verdict": "...", "rationale": "...",
  "blocking_issues": [...], "non_blocking_issues": [...] }
```

## 7. Sample workflow

```bash
# End of sprint 3
git fetch origin
python -m oss.gaussian.review.run \
    --sprint 3 \
    --commit-range origin/main..HEAD

# Exit 0 -> sprint 3 closes, sprint 4 begins.
# Exit 1 -> author addresses blocking_issues from judge.json, re-pushes,
#          re-runs the same command.
# Exit 3 -> escalation: human reviews judge.json + both reports.
```

Artifacts land in `oss/gaussian/review/artifacts/sprint-3/`. They are
checked in (small JSON files) so the audit trail survives across
machines.

## 8. What is NOT in v0

- No actual subagent dispatch -- `run.py` has a `dispatch_fn = None`
  placeholder. The next agent wires it to Claude Agent SDK or the
  `claude` CLI. The interface is tiny: `Callable[[str, str], str]`
  taking `(system_prompt, user_prompt)` and returning raw text.
- No automatic re-run loop on `REQUEST_CHANGES` -- the author drives
  the loop manually for now. A wrapper script can be added later.
- No cross-sprint trend analysis. Each sprint is judged in isolation.

## 9. File map

| File                                                | Purpose                                    |
|-----------------------------------------------------|--------------------------------------------|
| `oss/gaussian/review/run.py`                        | CLI entrypoint, git plumbing, orchestration |
| `oss/gaussian/review/reviewers.py`                  | A + B prompts and dispatch                 |
| `oss/gaussian/review/judge.py`                      | Judge prompt and dispatch                  |
| `oss/gaussian/review/schema.py`                     | Dataclass schemas + JSON I/O               |
| `oss/gaussian/review/README.md`                     | Operator-facing quick start                |
| `oss/gaussian/review/artifacts/sprint-N/*.json`     | Per-sprint review artifacts (generated)    |
