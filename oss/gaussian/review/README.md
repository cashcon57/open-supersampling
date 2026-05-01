# OSS-Gaussian Sprint Review Pipeline

Two-reviewer + one-judge gate that runs at the end of every OSS-Gaussian
sprint (7 sprints total). A sprint cannot close until the judge returns
`APPROVE`.

## Quick start

```bash
# Dry run -- DEFAULT (stub reviewers, heuristic judge, no LLM calls)
python -m oss.gaussian.review.run \
    --sprint 1 \
    --commit-range origin/main..HEAD \
    --dry-run

# Live run with real Claude API calls (opt-in)
pip install -e .[review]                       # one-time: install anthropic SDK
export ANTHROPIC_API_KEY="sk-ant-..."           # required for --use-api

python -m oss.gaussian.review.run \
    --sprint 1 \
    --commit-range origin/main..HEAD \
    --use-api
```

`--use-api` is **opt-in**. Without it the pipeline runs in dry-run mode
(no network, no cost) so you can smoke-test wiring safely. With it,
`oss/gaussian/review/dispatch.py` calls Claude `claude-sonnet-4-6` with
prompt caching on the system prompt and exponential backoff on
rate-limit / overload errors. Per-call token usage is logged to stderr.

Exit codes:

| Code | Verdict           | Meaning                                  |
|------|-------------------|------------------------------------------|
| 0    | `APPROVE`         | Sprint closes                            |
| 1    | `REQUEST_CHANGES` | Author fixes findings, pipeline re-runs  |
| 3    | `BLOCK`           | Escalates to human user                  |
| 2    | (error)           | Pipeline itself failed (bad range, etc.) |

## Artifacts

Each run writes four files under
`oss/gaussian/review/artifacts/sprint-N/`:

- `diff.patch` -- the exact unified diff that was reviewed
- `reviewer-a.json` -- correctness / security / performance findings
- `reviewer-b.json` -- spec / tests / integration findings
- `judge.json` -- structured verdict + classified blocking issues

## Architecture

```
        +-------------------+
        |  run.py (CLI)     |
        +---------+---------+
                  |
     +------------+-------------+
     |                          |
     v                          v
+-----------+              +-----------+
| Reviewer A|              | Reviewer B|     (parallel-safe)
| correctness|             | spec+intg |
+-----+-----+              +-----+-----+
      |                          |
      |   reviewer-a.json        |   reviewer-b.json
      +--------------+-----------+
                     |
                     v
                +---------+
                |  Judge  |  reads both reports + diff
                +----+----+
                     |
                     v
                judge.json (APPROVE / REQUEST_CHANGES / BLOCK)
```

## Real subagent dispatch

Real dispatch is implemented in
[`dispatch.py`](dispatch.py) and selected via the `--use-api` CLI flag.

`claude_dispatch(system, user) -> str` matches the
`DispatchFn = Callable[[str, str], str]` interface that
`dispatch_reviewer()` and `dispatch_judge()` both accept. Behaviour:

- **Model**: `claude-sonnet-4-6`.
- **Prompt caching**: the system prompt is marked `cache_control: ephemeral`,
  so within a single pipeline run the second + third calls (reviewer B,
  judge) hit the prompt cache and cost ~10x less on the cached portion.
- **Retries**: 3 attempts with exponential backoff (10s, 20s, 40s) on
  HTTP 429 (rate limit) and 529 (overload). Other errors propagate.
- **Token logging**: every call writes input / output / cache-hit token
  counts to stderr.

### Requirements for `--use-api`

| Requirement            | How to satisfy                              |
|------------------------|---------------------------------------------|
| Anthropic SDK          | `pip install -e .[review]`                  |
| API key                | `export ANTHROPIC_API_KEY="sk-ant-..."`     |

If the SDK is missing or the key is unset, `--use-api` raises a clear
error pointing at the fix. Drop `--use-api` to fall back to dry-run.

## Spec loading

The pipeline expects sprint specs at::

    docs/superpowers/specs/oss-gaussian-sprint-{N}.md

If a spec is missing the pipeline still runs but reviewers receive a
placeholder; expect a lower-signal review.

## Re-running after `REQUEST_CHANGES`

Just re-invoke with the same arguments after pushing fixes. Artifacts
overwrite in place; if you want history, snapshot
`artifacts/sprint-N/` between runs.
