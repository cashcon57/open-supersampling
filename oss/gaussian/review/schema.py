"""JSON schemas for OSS-Gaussian sprint code review pipeline.

Schemas are stdlib-only (dataclasses + json) so this module imports cleanly
in any environment. Pydantic is intentionally avoided to keep the review
tool dependency-free.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class Severity(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Verdict(str, Enum):
    APPROVE = "APPROVE"
    REQUEST_CHANGES = "REQUEST_CHANGES"
    BLOCK = "BLOCK"


# Reviewer category vocabularies. Free-form strings are tolerated (judge will
# group them) but reviewers should prefer these so artifacts stay queryable.
REVIEWER_A_CATEGORIES = (
    "correctness",
    "security",
    "performance",
    "style",
    "memory",
    "concurrency",
)
REVIEWER_B_CATEGORIES = (
    "spec_adherence",
    "test_coverage",
    "edge_case",
    "integration_risk",
    "regression_risk",
)


@dataclass
class Finding:
    """A single review finding from reviewer A or B."""

    severity: Severity
    category: str
    file: str
    lines: str  # e.g. "L42-58" or "L7"
    issue: str
    suggested_fix: str

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["severity"] = self.severity.value
        return d

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Finding":
        return cls(
            severity=Severity(raw["severity"]),
            category=str(raw["category"]),
            file=str(raw["file"]),
            lines=str(raw["lines"]),
            issue=str(raw["issue"]),
            suggested_fix=str(raw["suggested_fix"]),
        )


@dataclass
class ReviewerReport:
    """A reviewer agent's full output: ordered list of findings + a summary."""

    reviewer_id: str  # "A" or "B"
    sprint: int
    summary: str
    findings: list[Finding] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "reviewer_id": self.reviewer_id,
            "sprint": self.sprint,
            "summary": self.summary,
            "findings": [f.to_dict() for f in self.findings],
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ReviewerReport":
        return cls(
            reviewer_id=str(raw["reviewer_id"]),
            sprint=int(raw["sprint"]),
            summary=str(raw.get("summary", "")),
            findings=[Finding.from_dict(f) for f in raw.get("findings", [])],
        )


@dataclass
class JudgeVerdict:
    """Judge agent's structured verdict over both reviewer reports."""

    verdict: Verdict
    rationale: str
    blocking_issues: list[Finding] = field(default_factory=list)
    non_blocking_issues: list[Finding] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict.value,
            "rationale": self.rationale,
            "blocking_issues": [f.to_dict() for f in self.blocking_issues],
            "non_blocking_issues": [f.to_dict() for f in self.non_blocking_issues],
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "JudgeVerdict":
        return cls(
            verdict=Verdict(raw["verdict"]),
            rationale=str(raw.get("rationale", "")),
            blocking_issues=[Finding.from_dict(f) for f in raw.get("blocking_issues", [])],
            non_blocking_issues=[
                Finding.from_dict(f) for f in raw.get("non_blocking_issues", [])
            ],
        )


def dump_json(obj: Any, path: str) -> None:
    """Write any schema dataclass (or dict) to ``path`` as pretty JSON."""
    if hasattr(obj, "to_dict"):
        obj = obj.to_dict()
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, sort_keys=False)
        fh.write("\n")


def load_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)
