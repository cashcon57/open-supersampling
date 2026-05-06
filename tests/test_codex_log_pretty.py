from __future__ import annotations

from scripts.codex_log_pretty import render_html


def test_render_html_wraps_result_diff_hunks_in_details() -> None:
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

    assert '<details class="codex-diff-block">' in html
    assert "diff: 2 lines added, 1 lines removed, in demo.py" in html
    assert '<span class="codex-diff-add">+new</span>' in html
    assert '<span class="codex-result">plain output</span>' in html


def test_render_html_leaves_non_diff_result_lines_inline() -> None:
    log = "\n".join(
        [
            "exec",
            "pytest -q",
            " succeeded in 2ms:",
            "3 passed",
            "no diff here",
        ]
    )

    html = render_html(log)

    assert "<details" not in html
    assert '<span class="codex-result">3 passed</span>' in html
    assert '<span class="codex-result">no diff here</span>' in html
