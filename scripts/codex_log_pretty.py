#!/usr/bin/env python3
"""Stream a codex-exec log file with ANSI colors + section dividers.

Usage:
  ./scripts/codex_log_pretty.py /tmp/codex-v6model-stage2.log
  tail -f /tmp/codex-v6model-stage2.log | ./scripts/codex_log_pretty.py

Sections:
  - HEADER (codex banner)         dim grey
  - USER PROMPT                   dim cyan
  - REASONING (codex says...)     bright cyan
  - EXEC (tool/shell call)        bright yellow
  - RESULT (succeeded / exited)   green or red header, body uncolored
                                  with diff +/- colored if present

By default suppresses MCP transport ERROR noise. Pass --keep-mcp to keep.

Pure stdlib.
"""
from __future__ import annotations

import argparse
import textwrap
import re
import sys


RESET = "\x1b[0m"
DIM = "\x1b[2m"
BOLD = "\x1b[1m"

FG = {
    "red": "\x1b[31m",
    "green": "\x1b[32m",
    "yellow": "\x1b[33m",
    "cyan": "\x1b[36m",
    "grey": "\x1b[90m",
    "bgreen": "\x1b[92m",
    "bred": "\x1b[91m",
    "byellow": "\x1b[93m",
    "bcyan": "\x1b[96m",
    "bmagenta": "\x1b[95m",
}


def color(text: str, *codes: str) -> str:
    return "".join(codes) + text + RESET


MCP_ERROR_RE = re.compile(r"ERROR rmcp::transport::worker:")
SUCCESS_RE = re.compile(r"^ succeeded in (\d+)ms:")
EXIT_RE = re.compile(r"^ exited (\d+) in (\d+)ms:")


# State machine modes.
HEADER = "HEADER"
PROMPT = "PROMPT"
REASON = "REASON"
EXEC = "EXEC"
RESULT = "RESULT"


def section(label: str, fg: str) -> str:
    bar = (
        color("\n┌─[ ", BOLD, FG[fg])
        + color(label, BOLD, FG[fg])
        + color(" ]", BOLD, FG[fg])
    )
    return bar


def colorize_diff_line(line: str) -> str:
    if not line:
        return line
    if line.startswith("+++"):
        return color(line, BOLD, FG["bgreen"])
    if line.startswith("---") and not line.startswith("----"):
        return color(line, BOLD, FG["bred"])
    if line.startswith("@@"):
        return color(line, BOLD, FG["bmagenta"])
    if line[:1] == "+":
        return color(line, FG["green"])
    if line[:1] == "-":
        return color(line, FG["red"])
    return line


class Pretty:
    def __init__(self, keep_mcp: bool):
        self.keep_mcp = keep_mcp
        self.mode = HEADER
        self.out = sys.stdout

    def _emit_section(self, label: str, fg: str) -> None:
        self.out.write(section(label, fg) + "\n")

    def feed(self, raw_line: str) -> None:
        # Drop MCP transport noise unless requested.
        if not self.keep_mcp and MCP_ERROR_RE.search(raw_line):
            return

        line = raw_line.rstrip("\n")

        # Universal mode-transition triggers (checked before per-mode rules).
        if line == "user" and self.mode == HEADER:
            self._emit_section("USER PROMPT", "cyan")
            self.mode = PROMPT
            return
        if line == "codex":
            self._emit_section("REASONING", "bcyan")
            self.mode = REASON
            return
        if line == "exec":
            self._emit_section("EXEC", "byellow")
            self.mode = EXEC
            return

        # Result delimiters: appear after an exec call's command line.
        m_ok = SUCCESS_RE.match(line)
        m_err = EXIT_RE.match(line)
        if m_ok and self.mode in (EXEC, RESULT):
            label = "RESULT (ok in %sms)" % m_ok.group(1)
            self._emit_section(label, "green")
            self.mode = RESULT
            return
        if m_err and self.mode in (EXEC, RESULT):
            label = "RESULT (exit %s in %sms)" % (m_err.group(1), m_err.group(2))
            self._emit_section(label, "red")
            self.mode = RESULT
            return

        # Per-mode rendering.
        if self.mode == HEADER:
            self.out.write(color(line, DIM, FG["grey"]) + "\n")
        elif self.mode == PROMPT:
            self.out.write(color(line, DIM, FG["cyan"]) + "\n")
        elif self.mode == REASON:
            self.out.write(color(line, FG["bcyan"]) + "\n")
        elif self.mode == EXEC:
            self.out.write(color(line, FG["byellow"]) + "\n")
        elif self.mode == RESULT:
            self.out.write(colorize_diff_line(line) + "\n")
        else:
            self.out.write(line + "\n")
        self.out.flush()


def render_html(text: str, keep_mcp: bool = False) -> str:
    """Convert a codex-exec log body into an HTML fragment.

    Returns one big string of styled block/span elements for the OSS training
    dashboard, which provides the matching CSS classes.
    """
    import html as _html

    out: list[str] = []
    pending_diff: list[str] = []
    mode = HEADER

    def html_escape(s: str) -> str:
        return _html.escape(s, quote=False)

    def line_class_for_mode(m: str) -> str:
        return {
            HEADER: "codex-header",
            PROMPT: "codex-prompt",
            REASON: "codex-reason",
            EXEC: "codex-exec",
            RESULT: "codex-result",
        }.get(m, "codex-other")

    def diff_html(line: str) -> str:
        esc = html_escape(line)
        if line.startswith("+++"):
            return f'<span class="codex-diff-add-hd">{esc}</span>'
        if line.startswith("---") and not line.startswith("----"):
            return f'<span class="codex-diff-rm-hd">{esc}</span>'
        if line.startswith("@@"):
            return f'<span class="codex-diff-hunk">{esc}</span>'
        if line[:1] == "+":
            return f'<span class="codex-diff-add">{esc}</span>'
        if line[:1] == "-":
            return f'<span class="codex-diff-rm">{esc}</span>'
        return f'<span class="codex-result">{esc}</span>'

    def is_diff_line(line: str) -> bool:
        return line.startswith("@@") or line[:1] in ("+", "-")

    def diff_file(lines: list[str]) -> str:
        for line in lines:
            m = re.match(r"^\+\+\+ b/(.+)$", line)
            if m:
                return m.group(1)
        return "(unnamed diff)"

    def flush_diff() -> None:
        if not pending_diff:
            return
        added = sum(
            1 for line in pending_diff
            if line.startswith("+") and not line.startswith("+++")
        )
        removed = sum(
            1 for line in pending_diff
            if line.startswith("-") and not line.startswith("---")
        )
        fname = html_escape(diff_file(pending_diff))
        summary = f"diff: {added} lines added, {removed} lines removed, in {fname}"
        body = "\n".join(diff_html(line) for line in pending_diff)
        out.append(
            '<details class="codex-diff-block">'
            f'<summary><span class="codex-diff-carat">▶</span> {summary}</summary>'
            f'<div class="codex-diff-body">{body}</div>'
            '</details>'
        )
        pending_diff.clear()

    def reason_lines(line: str) -> list[str]:
        if not line:
            return [""]
        return textwrap.wrap(
            line,
            width=100,
            break_long_words=False,
            break_on_hyphens=False,
            replace_whitespace=False,
            drop_whitespace=True,
        ) or [line]

    def section_html(label: str, kind: str) -> str:
        return (
            f'<div class="codex-section-header codex-section-{kind}">'
            '<span class="codex-section-caret">▾</span>'
            '<span class="codex-section-bar"></span>'
            f'<span class="codex-section-icon" aria-hidden="true">{section_icon(kind)}</span>'
            f'<span class="codex-section-label">{html_escape(label)}</span>'
            '</div>'
        )

    def section_icon(kind: str) -> str:
        return {
            "prompt": "user",
            "reason": "think",
            "exec": "$",
            "ok": "ok",
            "err": "!",
        }.get(kind, "log")

    for raw in text.splitlines():
        if not keep_mcp and MCP_ERROR_RE.search(raw):
            continue
        line = raw

        # Mode transitions.
        if line == "user" and mode == HEADER:
            flush_diff()
            out.append(section_html("USER PROMPT", "prompt"))
            mode = PROMPT
            continue
        if line == "codex":
            flush_diff()
            out.append(section_html("REASONING", "reason"))
            mode = REASON
            continue
        if line == "exec":
            flush_diff()
            out.append(section_html("EXEC", "exec"))
            mode = EXEC
            continue

        m_ok = SUCCESS_RE.match(line)
        m_err = EXIT_RE.match(line)
        if m_ok and mode in (EXEC, RESULT):
            flush_diff()
            out.append(section_html(f"RESULT (ok in {m_ok.group(1)}ms)", "ok"))
            mode = RESULT
            continue
        if m_err and mode in (EXEC, RESULT):
            flush_diff()
            out.append(
                section_html(
                    f"RESULT (exit {m_err.group(1)} in {m_err.group(2)}ms)",
                    "err",
                )
            )
            mode = RESULT
            continue

        if mode == RESULT:
            if is_diff_line(line):
                pending_diff.append(line)
            else:
                flush_diff()
                out.append(diff_html(line))
        elif mode == REASON:
            for wrapped in reason_lines(line):
                out.append(f'<span class="codex-reason">{html_escape(wrapped)}</span>')
        elif mode == EXEC:
            out.append(f'<code class="cmd">{html_escape(line)}</code>')
        else:
            cls = line_class_for_mode(mode)
            out.append(f'<span class="{cls}">{html_escape(line)}</span>')

    flush_diff()
    return "\n".join(out)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("path", nargs="?",
                   help="codex log file (omit or '-' to read stdin)")
    p.add_argument("--keep-mcp", action="store_true",
                   help="Don't strip MCP transport ERROR noise.")
    args = p.parse_args()

    pretty = Pretty(keep_mcp=args.keep_mcp)
    if args.path is None or args.path == "-":
        for line in sys.stdin:
            pretty.feed(line)
    else:
        with open(args.path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                pretty.feed(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
