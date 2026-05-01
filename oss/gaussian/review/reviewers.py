"""Reviewer A and Reviewer B prompt templates + dispatch.

Two non-overlapping reviewer agents:

* **Reviewer A** -- implementation correctness: code correctness, security
  (path validation, no injection), performance (CUDA kernel efficiency,
  memory bandwidth, hot loops), idiomatic Python/CUDA/C++ style.
* **Reviewer B** -- spec adherence + integration: does the code do what the
  spec says, test coverage, edge cases, integration risks with existing OSS
  pixel-based code, regression risks.

Each reviewer is dispatched as a Claude subagent. This module owns the
prompt construction; the actual transport (Claude Agent SDK, raw API, or
``claude`` CLI) is intentionally pluggable -- see
:func:`dispatch_reviewer`.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Callable

from .schema import (
    REVIEWER_A_CATEGORIES,
    REVIEWER_B_CATEGORIES,
    Finding,
    ReviewerReport,
    Severity,
)

log = logging.getLogger(__name__)


REVIEWER_A_PROMPT = """\
You are **Reviewer A** in the OSS-Gaussian sprint review pipeline.

## Your concern: Implementation correctness
Review ONLY the following dimensions. Do NOT comment on spec adherence,
test coverage, or integration risks -- those belong to Reviewer B.

1. **Correctness** -- logic bugs, off-by-one, wrong math, undefined behaviour.
2. **Security** -- path validation (no traversal, no null bytes), no shell
   injection, no unchecked deserialization, no untrusted-input -> ``eval``.
3. **Performance** -- CUDA kernel efficiency (occupancy, divergence,
   shared-memory bank conflicts), memory bandwidth (coalescing, redundant
   copies), hot loops in Python (avoid per-pixel Python).
4. **Idiomatic style** -- Python (PEP 8, type hints, no mutable defaults),
   CUDA (``__restrict__``, ``constexpr``, launch bounds), C++ (RAII, no
   raw ``new``/``delete`` in new code).

## Input
You will receive:
- A unified diff for sprint {sprint}.
- The sprint spec (so you know what the code is *supposed* to do, but
  spec-adherence judgements are NOT yours -- only correctness within the
  stated intent).

## Output
Respond with a SINGLE JSON object, no prose, matching this schema:

```json
{{
  "reviewer_id": "A",
  "sprint": {sprint},
  "summary": "<one paragraph: overall correctness / perf / security posture>",
  "findings": [
    {{
      "severity": "high|medium|low",
      "category": "{categories}",
      "file": "<repo-relative path>",
      "lines": "L<start>-L<end> or L<n>",
      "issue": "<what is wrong>",
      "suggested_fix": "<concrete fix, code snippet ok>"
    }}
  ]
}}
```

Severity rubric:
- **high** -- ships a bug, security hole, or >10% perf regression.
- **medium** -- correctness smell, missing bounds check, suboptimal kernel.
- **low** -- style, naming, minor idiom.

If there are zero findings, return an empty ``findings`` list.
"""


REVIEWER_B_PROMPT = """\
You are **Reviewer B** in the OSS-Gaussian sprint review pipeline.

## Your concern: Spec adherence + integration
Review ONLY the following dimensions. Do NOT comment on low-level
correctness, performance, or style -- those belong to Reviewer A.

1. **Spec adherence** -- does the code implement what the sprint spec
   asks for? Missing acceptance criteria? Extra scope creep?
2. **Test coverage** -- are new code paths tested? Are the tests
   meaningful (assert behaviour, not just ``assert True``)?
3. **Edge cases** -- empty inputs, NaN/Inf in float tensors, zero-sized
   tiles, single-frame sequences, mixed precision.
4. **Integration risk** -- does this break existing OSS pixel-based
   code paths (OSS-RG, OSS-SR, OSS-Pico, OSS-FX)? Shared utility
   changes? Module boundary violations into ``oss/gaussian/``.
5. **Regression risk** -- changes to public APIs, on-disk formats,
   tensor shapes, or CLI flags that would break callers.

## Input
You will receive:
- A unified diff for sprint {sprint}.
- The sprint spec.
- A short summary of touched modules outside ``oss/gaussian/``.

## Output
Same JSON schema as Reviewer A but with ``"reviewer_id": "B"`` and
``category`` drawn from: {categories}.

Severity rubric:
- **high** -- spec acceptance criterion missing OR breaks existing
  pixel-based pipeline.
- **medium** -- weak test coverage on a new code path, plausible edge
  case unaddressed.
- **low** -- nit on naming consistency with existing OSS modules,
  doc gaps.
"""


@dataclass
class ReviewContext:
    """Inputs every reviewer agent needs."""

    sprint: int
    diff: str
    spec_text: str
    touched_modules_outside_gaussian: list[str]


# Type alias for the dispatch hook the next agent will wire up.
DispatchFn = Callable[[str, str], str]
"""(system_prompt, user_prompt) -> raw_model_output_string."""


def build_reviewer_a_prompt(ctx: ReviewContext) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for Reviewer A."""
    system = REVIEWER_A_PROMPT.format(
        sprint=ctx.sprint,
        categories="|".join(REVIEWER_A_CATEGORIES),
    )
    user = (
        f"## Sprint {ctx.sprint} spec\n\n{ctx.spec_text}\n\n"
        f"## Diff\n\n```diff\n{ctx.diff}\n```\n"
    )
    return system, user


def build_reviewer_b_prompt(ctx: ReviewContext) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for Reviewer B."""
    system = REVIEWER_B_PROMPT.format(
        sprint=ctx.sprint,
        categories="|".join(REVIEWER_B_CATEGORIES),
    )
    touched = "\n".join(f"- {m}" for m in ctx.touched_modules_outside_gaussian) or "- (none)"
    user = (
        f"## Sprint {ctx.sprint} spec\n\n{ctx.spec_text}\n\n"
        f"## Touched modules outside oss/gaussian/\n\n{touched}\n\n"
        f"## Diff\n\n```diff\n{ctx.diff}\n```\n"
    )
    return system, user


def parse_reviewer_response(raw: str, expected_id: str, sprint: int) -> ReviewerReport:
    """Parse a model's raw JSON output into a :class:`ReviewerReport`.

    Tolerates a Markdown ```json ... ``` fence around the payload.
    """
    text = raw.strip()
    if text.startswith("```"):
        # strip first fence line and trailing fence
        first_nl = text.find("\n")
        text = text[first_nl + 1 :] if first_nl != -1 else text
        if text.rstrip().endswith("```"):
            text = text.rstrip()[: -3]
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Reviewer {expected_id} did not return valid JSON: {exc}\n"
            f"--- raw output ---\n{raw[:500]}"
        ) from exc

    if payload.get("reviewer_id") != expected_id:
        raise ValueError(
            f"Reviewer ID mismatch: got {payload.get('reviewer_id')!r}, "
            f"expected {expected_id!r}"
        )
    if int(payload.get("sprint", -1)) != sprint:
        raise ValueError(
            f"Sprint mismatch: got {payload.get('sprint')}, expected {sprint}"
        )
    return ReviewerReport.from_dict(payload)


def dispatch_reviewer(
    reviewer_id: str,
    ctx: ReviewContext,
    dispatch: DispatchFn | None = None,
) -> ReviewerReport:
    """Run a reviewer agent.

    If ``dispatch`` is ``None`` we return a stub report so dry-runs work
    without an LLM. The next agent should pass a real :class:`DispatchFn`
    that calls Claude Agent SDK / CLI.
    """
    if reviewer_id == "A":
        system, user = build_reviewer_a_prompt(ctx)
    elif reviewer_id == "B":
        system, user = build_reviewer_b_prompt(ctx)
    else:
        raise ValueError(f"Unknown reviewer_id {reviewer_id!r}; expected 'A' or 'B'")

    if dispatch is None:
        log.warning("dispatch_reviewer: no dispatch fn -> returning DRY-RUN stub")
        return ReviewerReport(
            reviewer_id=reviewer_id,
            sprint=ctx.sprint,
            summary=f"DRY-RUN stub for reviewer {reviewer_id} (no LLM dispatch wired)",
            findings=[
                Finding(
                    severity=Severity.LOW,
                    category="dry_run",
                    file="<n/a>",
                    lines="L0",
                    issue="No dispatch function configured",
                    suggested_fix="Pass a DispatchFn into run_pipeline()",
                )
            ],
        )

    raw = dispatch(system, user)
    return parse_reviewer_response(raw, expected_id=reviewer_id, sprint=ctx.sprint)
