"""Real Anthropic SDK dispatch for the OSS-Gaussian review pipeline.

Implements the :data:`DispatchFn` interface
``Callable[[str, str], str]`` — ``(system_prompt, user_prompt) -> response_text``
— used by :mod:`oss.gaussian.review.reviewers` and
:mod:`oss.gaussian.review.judge`.

Design notes:

* **Model**: ``claude-sonnet-4-6`` — current Sonnet generation.
* **Prompt caching**: the system prompt is static across all sprints
  (only the sprint number is templated in), and within a sprint it is
  identical for both reviewer-A and reviewer-B re-runs. We mark it with
  ``cache_control: {"type": "ephemeral"}`` so the second + third call in
  a single pipeline run hit the prompt cache (~5 min TTL, ~10x cheaper
  on hits).
* **Retries**: 3 attempts with exponential backoff (10s base) on
  rate-limit (429) and overload (529) errors. Other errors propagate.
* **Token logging**: input / output / cache-hit token counts are written
  to stderr after every call so the operator can see cost as they go.

Opt-in: ``run.py --use-api``. Default remains dry-run.
"""

from __future__ import annotations

import os
import sys
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - import only for type checkers
    from anthropic import Anthropic

MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 4096
MAX_RETRIES = 3
BACKOFF_BASE_SEC = 10.0

_CLIENT: "Anthropic | None" = None


def _client() -> "Anthropic":
    """Lazy-init a single Anthropic client per process."""
    global _CLIENT
    if _CLIENT is not None:
        return _CLIENT
    try:
        from anthropic import Anthropic
    except ImportError as exc:
        raise RuntimeError(
            "anthropic SDK not installed. Run `pip install -e .[review]` "
            "(or `pip install anthropic`) to enable --use-api."
        ) from exc
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Export it in your shell, e.g.:\n"
            '  export ANTHROPIC_API_KEY="sk-ant-..."\n'
            "Or omit --use-api to run in dry-run (heuristic) mode."
        )
    _CLIENT = Anthropic()
    return _CLIENT


def _log_usage(usage: object) -> None:
    """Print token-usage line to stderr (best-effort)."""
    inp = getattr(usage, "input_tokens", None)
    out = getattr(usage, "output_tokens", None)
    cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
    cache_write = getattr(usage, "cache_creation_input_tokens", 0) or 0
    print(
        f"[claude_dispatch] tokens: in={inp} out={out} "
        f"cache_read={cache_read} cache_write={cache_write}",
        file=sys.stderr,
    )


def claude_dispatch(system: str, user: str) -> str:
    """Send (system, user) to Claude and return assistant text.

    Matches :data:`oss.gaussian.review.reviewers.DispatchFn`.
    Caches the system prompt; retries on 429/529 with exponential backoff.
    """
    from anthropic import APIStatusError, RateLimitError

    client = _client()
    # System block as a list so we can attach cache_control.
    system_blocks = [
        {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}
    ]
    messages = [{"role": "user", "content": user}]

    last_exc: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                system=system_blocks,
                messages=messages,
            )
            _log_usage(resp.usage)
            # First text block is the assistant's reply.
            for block in resp.content:
                if getattr(block, "type", None) == "text":
                    return block.text
            raise RuntimeError("Claude returned no text block")
        except (RateLimitError, APIStatusError) as exc:
            status = getattr(exc, "status_code", None)
            # Retry on 429 (rate limit) and 529 (overloaded); else re-raise.
            if status not in (429, 529):
                raise
            last_exc = exc
            if attempt == MAX_RETRIES - 1:
                break
            sleep_for = BACKOFF_BASE_SEC * (2**attempt)
            print(
                f"[claude_dispatch] {status} on attempt {attempt + 1}/"
                f"{MAX_RETRIES}; sleeping {sleep_for:.0f}s",
                file=sys.stderr,
            )
            time.sleep(sleep_for)
    raise RuntimeError(
        f"claude_dispatch: exhausted {MAX_RETRIES} retries"
    ) from last_exc
