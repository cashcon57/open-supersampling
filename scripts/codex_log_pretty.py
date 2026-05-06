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
# Codex's ``apply_patch`` tool prints these markers on a bare line each.
# The state machine treats them like an exec action so the patch body
# is collapsed inside <details> and not rendered inline as reasoning.
APPLY_PATCH_OPEN = "apply patch"
APPLY_PATCH_OK = "patch: completed"
APPLY_PATCH_FAIL_RE = re.compile(r"^patch: (failed|error)")


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
    """Render a codex-exec log body as HTML that matches the real codex
    TUI as closely as possible inside a single ``<pre>`` container.

    Layout principles:
      * Reasoning + user prompts flow as plain colored text — no per-line
        boxes, no per-line decorators. Newlines come from the source.
      * Each ``exec`` + matching ``succeeded/exited`` block collapses into
        a single ``<details>`` whose summary is ``$ <command>`` plus the
        result status. The full result body lives in the expanded view.
      * Diff hunks inside an action body get +/- coloring inline (still
        plain text, no boxes).
      * The dashboard CSS turns the wrapping ``<pre>`` into the scrollable
        viewport; this renderer just emits inline spans + the action
        ``<details>`` blocks.
    """
    import html as _html

    out: list[str] = []

    def esc(s: str) -> str:
        return _html.escape(s, quote=False)

    def color_diff_line(line: str) -> str:
        e = esc(line)
        if line.startswith("+++"):
            return f'<span class="codex-diff-add-hd">{e}</span>'
        if line.startswith("---") and not line.startswith("----"):
            return f'<span class="codex-diff-rm-hd">{e}</span>'
        if line.startswith("@@"):
            return f'<span class="codex-diff-hunk">{e}</span>'
        if line[:1] == "+":
            return f'<span class="codex-diff-add">{e}</span>'
        if line[:1] == "-":
            return f'<span class="codex-diff-rm">{e}</span>'
        return e

    # --- action accumulation (exec + result rolled into one <details>) ---
    action_cmd_lines: list[str] = []   # everything between `exec` and the
                                       # ` succeeded/exited` marker.
    action_status: Optional[str] = None  # "ok in 12ms" / "exit 1 in 5ms"
    action_status_class: str = "ok"
    action_body_lines: list[str] = []  # everything after the marker.
    in_exec = False  # accumulating command lines
    in_result = False  # accumulating result body

    def open_action() -> None:
        nonlocal in_exec, in_result
        nonlocal action_cmd_lines, action_status, action_status_class, action_body_lines
        action_cmd_lines = []
        action_status = None
        action_status_class = "ok"
        action_body_lines = []
        in_exec = True
        in_result = False

    def close_action() -> None:
        nonlocal in_exec, in_result
        if not (in_exec or in_result):
            return
        # Action block: <details> with summary line + collapsed body.
        # The body is hidden until the user clicks the summary, like
        # real codex / Claude Code. Default state is collapsed —
        # the conversation flows through prompts + reasoning, with
        # tool-call output one click away when needed.
        cmd = " ".join(s.strip() for s in action_cmd_lines if s.strip()).strip()
        if not cmd:
            cmd = "(empty exec)"
        # Collapse multi-shell prefixes for legibility:
        #   "/bin/zsh -lc 'sed -n 1,40p file'" -> "sed -n 1,40p file"
        m = re.match(r"^/bin/zsh -lc ['\"](.+)['\"](?:\s+in\s+.+)?$", cmd)
        if m:
            cmd = m.group(1)
        if len(cmd) > 200:
            cmd = cmd[:197] + "..."
        status_txt = action_status or "running"
        n_body = len(action_body_lines)
        body_hint = (
            f" · {n_body} line{'s' if n_body != 1 else ''} hidden"
            if n_body
            else ""
        )
        summary = (
            f'<span class="codex-action-prompt">$</span> '
            f'<span class="codex-action-cmd">{esc(cmd)}</span>'
            f'<span class="codex-action-status codex-status-{action_status_class}">'
            f' {esc(status_txt)}{esc(body_hint)}</span>'
        )
        body_html = "\n".join(color_diff_line(b) for b in action_body_lines)
        # IMPORTANT: <details> collapses its non-<summary> children by
        # default (HTML spec). The dashboard CSS does not override this
        # — clicking the summary toggles open/closed, identical to the
        # codex-CLI / Claude Code action-collapse UX.
        out.append(
            '<details class="codex-action">'
            f'<summary>{summary}</summary>'
            f'<pre class="codex-action-body">{body_html}</pre>'
            '</details>'
        )
        in_exec = False
        in_result = False

    mode = HEADER
    for raw in text.splitlines():
        if not keep_mcp and MCP_ERROR_RE.search(raw):
            continue
        line = raw

        # Mode transitions: an action (exec block) gets closed and emitted
        # the moment we see any non-action marker.
        if line == "user" and mode == HEADER:
            close_action()
            out.append('<span class="codex-mode-label">user prompt:</span>')
            mode = PROMPT
            continue
        if line == "codex":
            close_action()
            out.append('<span class="codex-mode-label">codex:</span>')
            mode = REASON
            continue
        if line == "exec":
            close_action()
            open_action()
            mode = EXEC
            continue
        if line == APPLY_PATCH_OPEN:
            # Treat `apply patch` as a synthetic exec block. Codex omits
            # the standard "exec\n<command>" prefix for this tool, but
            # the body that follows is still tool-call output (file
            # paths + diff hunks) that belongs in a collapsed <details>.
            close_action()
            open_action()
            action_cmd_lines.append("apply_patch")
            mode = EXEC
            continue

        m_ok = SUCCESS_RE.match(line)
        m_err = EXIT_RE.match(line)
        if m_ok and mode in (EXEC, RESULT):
            action_status = f"✓ ok in {m_ok.group(1)}ms"
            action_status_class = "ok"
            in_exec = False
            in_result = True
            mode = RESULT
            continue
        if m_err and mode in (EXEC, RESULT):
            action_status = f"✗ exit {m_err.group(1)} in {m_err.group(2)}ms"
            action_status_class = "err"
            in_exec = False
            in_result = True
            mode = RESULT
            continue
        if line == APPLY_PATCH_OK and mode in (EXEC, RESULT):
            action_status = "✓ patch applied"
            action_status_class = "ok"
            in_exec = False
            in_result = True
            mode = RESULT
            continue
        if APPLY_PATCH_FAIL_RE.match(line) and mode in (EXEC, RESULT):
            action_status = f"✗ {line}"
            action_status_class = "err"
            in_exec = False
            in_result = True
            mode = RESULT
            continue

        if mode == RESULT:
            action_body_lines.append(line)
        elif mode == EXEC:
            action_cmd_lines.append(line)
        elif mode == REASON:
            out.append(f'<span class="codex-reason">{esc(line)}</span>')
        elif mode == PROMPT:
            out.append(f'<span class="codex-prompt">{esc(line)}</span>')
        elif mode == HEADER:
            out.append(f'<span class="codex-header">{esc(line)}</span>')

    close_action()
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
