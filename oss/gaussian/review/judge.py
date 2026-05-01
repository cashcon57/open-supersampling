"""Judge agent: consumes both reviewer reports + the diff, emits verdict.

The judge is a third subagent. Its job is NOT to re-review the code; it is
to weigh the two reviewer reports against the diff and decide whether the
sprint can close.
"""

from __future__ import annotations

import json
import logging
from typing import Callable

from .schema import Finding, JudgeVerdict, ReviewerReport, Verdict

log = logging.getLogger(__name__)


JUDGE_PROMPT = """\
You are the **Judge** in the OSS-Gaussian sprint review pipeline.

You are NOT a reviewer. Two reviewers (A: implementation correctness,
B: spec adherence + integration) have already submitted JSON findings.
Your job is to weigh their findings against the diff and decide whether
sprint {sprint} can close.

## Decision rubric

- **APPROVE** -- no high-severity findings from either reviewer, and at
  most a small number of mediums that are non-blocking style/test gaps.
  Sprint closes; non-blocking issues are filed for the next sprint.
- **REQUEST_CHANGES** -- one or more medium/high findings that are
  fixable in <= 1 day of work. Sprint loops back to author; pipeline
  re-runs after fixes.
- **BLOCK** -- a high-severity correctness, security, or spec-adherence
  finding that requires a design change, OR the two reviewers disagree
  about something material that you cannot resolve from the diff.
  Escalates to the human user.

## Conflict handling
If reviewer A says "performance is fine" and reviewer B says "this
breaks the existing pipeline", the integration concern wins -- never
let perf optimism override a regression risk. Conversely, if reviewer
B says "spec satisfied" but reviewer A flags a high-severity
correctness bug, the bug wins.

## Output
Respond with a SINGLE JSON object, no prose:

```json
{{
  "verdict": "APPROVE|REQUEST_CHANGES|BLOCK",
  "rationale": "<2-4 sentences: why this verdict>",
  "blocking_issues":     [ <Finding>, ... ],
  "non_blocking_issues": [ <Finding>, ... ]
}}
```

Each ``Finding`` is copied verbatim from one of the reviewer reports
(same fields: severity, category, file, lines, issue, suggested_fix).
Re-classify into blocking vs non-blocking per your rubric -- you do
NOT have to honour the reviewer's severity if you disagree, but say
so in ``rationale``.
"""


DispatchFn = Callable[[str, str], str]


def build_judge_prompt(
    sprint: int,
    diff: str,
    report_a: ReviewerReport,
    report_b: ReviewerReport,
) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for the judge."""
    system = JUDGE_PROMPT.format(sprint=sprint)
    user = (
        f"## Reviewer A report\n\n```json\n"
        f"{json.dumps(report_a.to_dict(), indent=2)}\n```\n\n"
        f"## Reviewer B report\n\n```json\n"
        f"{json.dumps(report_b.to_dict(), indent=2)}\n```\n\n"
        f"## Diff (for context only -- do not re-review)\n\n"
        f"```diff\n{diff}\n```\n"
    )
    return system, user


def parse_judge_response(raw: str) -> JudgeVerdict:
    """Parse the judge's raw output JSON into a :class:`JudgeVerdict`."""
    text = raw.strip()
    if text.startswith("```"):
        first_nl = text.find("\n")
        text = text[first_nl + 1 :] if first_nl != -1 else text
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Judge did not return valid JSON: {exc}\n"
            f"--- raw output ---\n{raw[:500]}"
        ) from exc
    return JudgeVerdict.from_dict(payload)


def _heuristic_verdict(
    report_a: ReviewerReport, report_b: ReviewerReport
) -> JudgeVerdict:
    """Stub verdict used when no LLM dispatch is wired (dry runs).

    Conservative rule: any high -> BLOCK; any medium -> REQUEST_CHANGES;
    else APPROVE.
    """
    all_findings: list[Finding] = list(report_a.findings) + list(report_b.findings)
    highs = [f for f in all_findings if f.severity.value == "high"]
    meds = [f for f in all_findings if f.severity.value == "medium"]
    lows = [f for f in all_findings if f.severity.value == "low"]

    if highs:
        verdict = Verdict.BLOCK
        rationale = f"Heuristic dry-run: {len(highs)} high-severity finding(s) -> BLOCK."
        blocking, non_blocking = highs + meds, lows
    elif meds:
        verdict = Verdict.REQUEST_CHANGES
        rationale = f"Heuristic dry-run: {len(meds)} medium finding(s) -> REQUEST_CHANGES."
        blocking, non_blocking = meds, lows
    else:
        verdict = Verdict.APPROVE
        rationale = "Heuristic dry-run: no medium/high findings -> APPROVE."
        blocking, non_blocking = [], lows

    return JudgeVerdict(
        verdict=verdict,
        rationale=rationale,
        blocking_issues=blocking,
        non_blocking_issues=non_blocking,
    )


def dispatch_judge(
    sprint: int,
    diff: str,
    report_a: ReviewerReport,
    report_b: ReviewerReport,
    dispatch: DispatchFn | None = None,
) -> JudgeVerdict:
    """Run the judge. Falls back to a heuristic verdict on dry-run."""
    if dispatch is None:
        log.warning("dispatch_judge: no dispatch fn -> heuristic verdict")
        return _heuristic_verdict(report_a, report_b)

    system, user = build_judge_prompt(sprint, diff, report_a, report_b)
    raw = dispatch(system, user)
    return parse_judge_response(raw)
