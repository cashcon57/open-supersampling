"""CLI entrypoint for the OSS-Gaussian sprint review pipeline.

Usage::

    python -m oss.gaussian.review.run --sprint 3 --commit-range origin/main..HEAD
    python -m oss.gaussian.review.run --sprint 3 --commit-range A..B --dry-run

The ``--dry-run`` flag runs the full pipeline with stub reviewers + a
heuristic judge, so the wiring can be smoke-tested before any real
subagent dispatch is plumbed in.

Artifacts are written to::

    oss/gaussian/review/artifacts/sprint-N/
        diff.patch
        reviewer-a.json
        reviewer-b.json
        judge.json
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

from .judge import dispatch_judge
from .reviewers import DispatchFn, ReviewContext, dispatch_reviewer
from .schema import JudgeVerdict, Verdict, dump_json

log = logging.getLogger("oss.gaussian.review")

REPO_ROOT = Path(__file__).resolve().parents[3]
"""Path to the open-reconstruction-suite checkout (parent of ``oss/``)."""

ARTIFACTS_ROOT = REPO_ROOT / "oss" / "gaussian" / "review" / "artifacts"
SPECS_ROOT = REPO_ROOT / "docs" / "superpowers" / "specs"


# ---------------------------------------------------------------------------
# git helpers
# ---------------------------------------------------------------------------

def _run_git(args: list[str]) -> str:
    """Run a git command in the repo and return stdout (raises on failure)."""
    result = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def get_diff(commit_range: str) -> str:
    """Return the unified diff for ``commit_range`` (e.g. ``A..B``)."""
    return _run_git(["diff", commit_range])


def get_touched_files(commit_range: str) -> list[str]:
    """Return repo-relative paths of files changed in ``commit_range``."""
    out = _run_git(["diff", "--name-only", commit_range])
    return [line for line in out.splitlines() if line.strip()]


def files_outside_gaussian(touched: list[str]) -> list[str]:
    """Subset of touched files that live outside ``oss/gaussian/``."""
    return [f for f in touched if not f.startswith("oss/gaussian/")]


# ---------------------------------------------------------------------------
# spec loading
# ---------------------------------------------------------------------------

def load_sprint_spec(sprint: int) -> str:
    """Load the sprint spec text.

    Convention: ``docs/superpowers/specs/oss-gaussian-sprint-N.md``. If the
    file is missing we return a placeholder so the pipeline can still run
    (the reviewers will say "no spec provided").
    """
    candidate = SPECS_ROOT / f"oss-gaussian-sprint-{sprint}.md"
    if candidate.is_file():
        return candidate.read_text(encoding="utf-8")
    log.warning("Sprint spec not found at %s -- using placeholder", candidate)
    return f"(No spec file found for sprint {sprint} at {candidate})"


# ---------------------------------------------------------------------------
# pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    sprint: int,
    commit_range: str,
    dry_run: bool = False,
    use_api: bool = False,
) -> JudgeVerdict:
    """Run the full A -> B -> judge pipeline. Returns the judge's verdict.

    Side effect: writes all four artifacts under
    ``oss/gaussian/review/artifacts/sprint-{sprint}/``.

    If ``use_api`` is True (and ``dry_run`` is False), the pipeline calls
    the real Anthropic API via :func:`oss.gaussian.review.dispatch.claude_dispatch`.
    Otherwise reviewers + judge fall back to their stub / heuristic
    implementations.
    """
    if not 1 <= sprint <= 7:
        raise ValueError(f"Sprint {sprint} out of range 1..7")

    artifact_dir = ARTIFACTS_ROOT / f"sprint-{sprint}"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    diff = get_diff(commit_range)
    if not diff.strip():
        raise RuntimeError(
            f"No diff for commit range {commit_range!r}; nothing to review."
        )
    (artifact_dir / "diff.patch").write_text(diff, encoding="utf-8")

    touched = get_touched_files(commit_range)
    outside = files_outside_gaussian(touched)
    spec_text = load_sprint_spec(sprint)

    ctx = ReviewContext(
        sprint=sprint,
        diff=diff,
        spec_text=spec_text,
        touched_modules_outside_gaussian=outside,
    )

    # Pick dispatch: real API (--use-api) or None (dry-run stubs).
    dispatch_fn: DispatchFn | None = None
    if use_api and not dry_run:
        from .dispatch import claude_dispatch  # local import: optional dep
        dispatch_fn = claude_dispatch
        log.info("Using real Anthropic API dispatch (claude-sonnet-4-6)")
    else:
        log.info("Dry-run mode: stub reviewers + heuristic judge")

    log.info("Dispatching Reviewer A (sprint %d)...", sprint)
    report_a = dispatch_reviewer("A", ctx, dispatch=dispatch_fn)
    dump_json(report_a, str(artifact_dir / "reviewer-a.json"))

    log.info("Dispatching Reviewer B (sprint %d)...", sprint)
    report_b = dispatch_reviewer("B", ctx, dispatch=dispatch_fn)
    dump_json(report_b, str(artifact_dir / "reviewer-b.json"))

    log.info("Dispatching Judge (sprint %d)...", sprint)
    verdict = dispatch_judge(sprint, diff, report_a, report_b, dispatch=dispatch_fn)
    dump_json(verdict, str(artifact_dir / "judge.json"))

    return verdict


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m oss.gaussian.review.run",
        description="Run the OSS-Gaussian sprint code review pipeline.",
    )
    p.add_argument("--sprint", type=int, required=True, help="Sprint number, 1..7")
    p.add_argument(
        "--commit-range",
        required=True,
        help="Git commit range to review, e.g. origin/main..HEAD",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Run with stub reviewers + heuristic judge (no LLM calls). "
             "This is the default when --use-api is not set.",
    )
    p.add_argument(
        "--use-api",
        action="store_true",
        help="Make real Anthropic API calls (requires ANTHROPIC_API_KEY "
             "and `pip install -e .[review]`). Off by default for safety.",
    )
    p.add_argument(
        "-v", "--verbose", action="store_true", help="Verbose logging"
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        verdict = run_pipeline(
            sprint=args.sprint,
            commit_range=args.commit_range,
            dry_run=args.dry_run,
            use_api=args.use_api,
        )
    except (ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        log.error("Pipeline failed: %s", exc)
        return 2

    print(f"\n=== Sprint {args.sprint} verdict: {verdict.verdict.value} ===")
    print(f"Rationale: {verdict.rationale}")
    print(f"Blocking issues:     {len(verdict.blocking_issues)}")
    print(f"Non-blocking issues: {len(verdict.non_blocking_issues)}")

    # Exit codes: 0 APPROVE, 1 REQUEST_CHANGES, 3 BLOCK -- so CI can gate.
    return {
        Verdict.APPROVE: 0,
        Verdict.REQUEST_CHANGES: 1,
        Verdict.BLOCK: 3,
    }[verdict.verdict]


if __name__ == "__main__":
    sys.exit(main())
