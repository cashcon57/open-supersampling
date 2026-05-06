from __future__ import annotations

from scripts.codex_log_pretty import render_html


def test_render_html_wraps_exec_plus_result_in_collapsed_action() -> None:
    """An ``exec`` block followed by ``succeeded`` collapses into a single
    ``<details class='codex-action'>``: summary shows ``$ <command>`` plus
    the status, body holds the full output."""
    log = "\n".join(
        [
            "exec",
            "git diff",
            " succeeded in 1ms:",
            "diff --git a/demo.py b/demo.py",
            "--- a/demo.py",
            "+++ b/demo.py",
            "@@ -1,2 +1,3 @@",
            "-old",
            "+new",
            "+extra",
            "plain output",
        ]
    )

    html = render_html(log)

    # Single collapsed action element wraps the whole exec + result.
    assert html.count('<details class="codex-action">') == 1
    # Summary: $ command + ✓ ok status.
    assert '<span class="codex-action-prompt">$</span>' in html
    assert "git diff" in html
    assert "✓ ok in 1ms" in html
    # Body is inside .codex-action-body and gets diff-line coloring inline.
    assert '<pre class="codex-action-body">' in html
    assert '<span class="codex-diff-add">+new</span>' in html
    assert '<span class="codex-diff-add-hd">+++ b/demo.py</span>' in html
    # Non-diff result lines stay as plain text inside the body.
    assert "plain output" in html


def test_render_html_collapses_zsh_wrapper_in_command_summary() -> None:
    """``/bin/zsh -lc 'real cmd'`` should show as ``real cmd`` in summary."""
    log = "\n".join(
        [
            "exec",
            "/bin/zsh -lc 'sed -n 1,10p file.py' in /Users/foo",
            " succeeded in 0ms:",
            "(file body)",
        ]
    )
    html = render_html(log)
    assert "sed -n 1,10p file.py" in html
    assert "/bin/zsh -lc" not in html  # collapsed away


def test_render_html_failed_exec_marks_status_err() -> None:
    log = "\n".join(
        [
            "exec",
            "false",
            " exited 1 in 5ms:",
            "no output",
        ]
    )
    html = render_html(log)
    assert "✗ exit 1 in 5ms" in html
    assert 'codex-status-err' in html


def test_render_html_reasoning_blocks_are_plain_inline() -> None:
    """REASONING lines render as plain colored spans; no boxes, no per-line
    structural decoration."""
    log = "\n".join(
        [
            "codex",
            "First I will read the prompt.",
            "Then dispatch a subagent.",
        ]
    )
    html = render_html(log)
    assert '<span class="codex-mode-label">codex:</span>' in html
    assert '<span class="codex-reason">First I will read the prompt.</span>' in html
    assert '<span class="codex-reason">Then dispatch a subagent.</span>' in html
    # No per-line boxes.
    assert "codex-section-header" not in html
    assert "codex-section-bar" not in html
