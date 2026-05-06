"""Tiny live dashboard for an OSS-Gaussian / OSS-SR training run.

Serves a single HTML page on http://<host>:<port>/ that polls JSON
endpoints for metrics, score history, and the tail of the log file.
Designed to read the rolling files written by `oss/gaussian/train/train.py`.

Run on the same machine as the training process:

    python scripts/training_dashboard.py \\
        --output-dir <train-host-data>/checkpoints/srcnn-prod-v3 \\
        --log-file <train-host-data>/logs/srcnn-prod-v3.log \\
        --port 8080 --host 0.0.0.0

Then point a browser at http://<this-machine-tailnet>:8080/ — works across
the tailnet because we bind to 0.0.0.0 by default.

No external dependencies beyond Python stdlib + a CDN-loaded Chart.js for
charts (the HTML pulls Chart.js from cdn.jsdelivr.net at page load).
"""

from __future__ import annotations

import argparse
import fnmatch
import html as _html
import json
import os
import re
import socketserver
import sys
import time
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlparse

# Importable HTML renderer for /tmp/codex-*.log content (lives next door).
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from codex_log_pretty import render_html as _render_codex_html  # type: ignore[import-not-found]
except Exception:  # pragma: no cover — optional, dashboard still works without
    _render_codex_html = None  # type: ignore[assignment]


CODEX_LOG_DIR = Path("/tmp")
CODEX_LOG_GLOB = "codex-*.log"
CODEX_ACTIVE_SECONDS = 60 * 60
CODEX_LOG_CAP_BYTES = 4 * 1024 * 1024
CODEX_STREAM_ENTRY_LIMIT = 160
CODEX_TIMESTAMP_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z)\b"
)


RUN_DIR_PATTERNS = (
    "srcnn-v*-temporal*",
    "srcnn-v6-*",
    "srcnn-v5-pixel-temporal*",
    "srcnn-v5-gaussian-temporal*",
)

# ---------------------------------------------------------------------------
# HTML page (single file, no build step). Chart.js is CDN-loaded.
# ---------------------------------------------------------------------------

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>OSS Training Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0"></script>
<!-- Pan + wheel-zoom for every chart (drag to pan x-axis, wheel to zoom on cursor). -->
<script src="https://cdn.jsdelivr.net/npm/hammerjs@2.0.8"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-zoom@2.0.1"></script>
<style>
  :root {
    --bg: #0e1116;
    --panel: #161b22;
    --border: #30363d;
    --fg: #e6edf3;
    --muted: #8b949e;
    --good: #3fb950;
    --warn: #d29922;
    --bad:  #f85149;
    --link: #58a6ff;
    --mono: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  }
  * { box-sizing: border-box; }
  body { margin: 0; padding: 24px; background: var(--bg); color: var(--fg);
         font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
  h1 { font-size: 20px; margin: 0 0 8px 0; }
  .sub { color: var(--muted); font-size: 13px; margin-bottom: 24px; }
  .topbar { display: flex; align-items: flex-start; justify-content: space-between;
            gap: 16px; margin-bottom: 24px; }
  .topbar .sub { margin-bottom: 0; }
  .run-picker { display: flex; align-items: center; gap: 8px; white-space: nowrap;
                color: var(--muted); font-size: 12px; }
  .run-picker select {
    background: #21262d; color: var(--fg); border: 1px solid var(--border);
    border-radius: 4px; padding: 5px 28px 5px 8px; font-family: var(--mono);
    font-size: 12px; max-width: min(52vw, 520px);
  }
  .grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; }
  .panel { background: var(--panel); border: 1px solid var(--border);
           border-radius: 8px; padding: 16px; }
  .panel h2 { font-size: 14px; margin: 0 0 12px 0; color: var(--muted);
              text-transform: uppercase; letter-spacing: 0.5px; }
  .panel-head { display: flex; justify-content: space-between; align-items: center;
                margin: 0 0 12px 0; gap: 12px; }
  .panel-head h2 { margin: 0; }
  .chart-toolbar { display: flex; gap: 4px; }
  .chart-toolbar button {
    background: #21262d; color: var(--fg); border: 1px solid var(--border);
    border-radius: 4px; padding: 2px 8px; font-size: 11px; cursor: pointer;
    font-family: var(--mono); min-width: 24px;
  }
  .chart-toolbar button:hover { background: #30363d; border-color: var(--link); }
  .chart-toolbar button:active { background: #0d1117; }
  .chart-legend { display: flex; flex-wrap: wrap; gap: 4px 12px;
                  margin-top: 8px; font-size: 11px; }
  .chart-legend label { display: inline-flex; align-items: center; gap: 4px;
                        cursor: pointer; user-select: none; color: var(--fg); }
  .chart-legend label.off { color: var(--muted); text-decoration: line-through; }
  .chart-legend input[type="checkbox"] { margin: 0; cursor: pointer; }
  .chart-legend .swatch { display: inline-block; width: 14px; height: 3px;
                          border-radius: 1px; }
  .stat-row { display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 16px; }
  .stat { flex: 1; min-width: 140px; }
  .stat .lbl { font-size: 11px; color: var(--muted); text-transform: uppercase; }
  .stat .val { font-size: 24px; font-weight: 600; font-family: var(--mono); }
  .stat .val.good { color: var(--good); }
  .stat .val.warn { color: var(--warn); }
  .stat .val.bad  { color: var(--bad); }
  .progress { height: 8px; background: #21262d; border-radius: 4px; overflow: hidden; }
  .progress-fill { height: 100%; background: var(--good); transition: width 0.3s ease; }
  .log { font-family: var(--mono); font-size: 12px; white-space: pre-wrap;
         background: #0d1117; border: 1px solid var(--border); border-radius: 4px;
         padding: 12px; max-height: 360px; overflow-y: auto; line-height: 1.5; }
  /* Codex log panel — match the real codex TUI: plain text inside a
     <pre>, no per-line boxes, action blocks (exec+result) collapsed
     into a one-line summary that expands to show the full output. */
  .codex-log {
    font-family: var(--mono); font-size: 12px; white-space: pre-wrap; word-break: normal;
    background: #0d1117; border: 1px solid var(--border); border-radius: 4px;
    padding: 12px; max-height: 560px; overflow-y: auto; line-height: 1.45;
    color: var(--fg); display: block; margin: 0;
  }
  .codex-log .codex-mode-label {
    color: var(--muted); font-weight: 700; letter-spacing: 0.2px;
    text-transform: lowercase;
  }
  .codex-log .codex-header { color: #6e7681; opacity: 0.7; }
  .codex-log .codex-prompt { color: #79c0ff; opacity: 0.85; }
  .codex-log .codex-reason { color: #c9d1d9; }
  /* Action block: <details> wrapping exec command + result body. */
  .codex-log .codex-action {
    margin: 0; padding: 0; display: block;
  }
  .codex-log .codex-action > summary {
    cursor: pointer; padding: 1px 0; user-select: none;
    list-style: none; white-space: pre-wrap; line-height: 1.5;
  }
  .codex-log .codex-action > summary::-webkit-details-marker { display: none; }
  .codex-log .codex-action > summary::before {
    content: "▶ "; color: var(--muted); font-size: 10px; margin-right: 2px;
  }
  .codex-log .codex-action[open] > summary::before { content: "▼ "; }
  .codex-log .codex-action-prompt { color: #d2a8ff; font-weight: 700; }
  .codex-log .codex-action-cmd    { color: #ffdf5d; }
  .codex-log .codex-action-status { color: var(--muted); font-size: 11px; margin-left: 6px; }
  .codex-log .codex-status-ok    { color: #3fb950; }
  .codex-log .codex-status-err   { color: #f85149; }
  .codex-log .codex-action-body {
    margin: 4px 0 6px 14px; padding: 6px 8px;
    background: #0f151c; border-left: 2px solid #27313a;
    color: var(--fg); font-family: var(--mono); font-size: 11px;
    white-space: pre; overflow-x: auto; max-height: 280px; overflow-y: auto;
  }
  .codex-log .codex-diff-add    { color: #56d364; }
  .codex-log .codex-diff-rm     { color: #ff7b72; }
  .codex-log .codex-diff-add-hd { color: #3fb950; font-weight: 700; }
  .codex-log .codex-diff-rm-hd  { color: #f85149; font-weight: 700; }
  .codex-log .codex-diff-hunk   { color: #d2a8ff; font-weight: 700; }
  .codex-toolbar { display: flex; flex-wrap: wrap; align-items: center; gap: 8px;
                   margin-bottom: 8px; font-size: 12px; color: var(--muted); }
  .codex-toolbar select {
    background: #21262d; color: var(--fg); border: 1px solid var(--border);
    border-radius: 4px; padding: 4px 8px; font-family: var(--mono); font-size: 12px;
    max-width: min(60vw, 480px);
  }
  .codex-toolbar label { display: inline-flex; align-items: center; gap: 4px; cursor: pointer; }
  .codex-toolbar .codex-file-filters { display: inline-flex; flex-wrap: wrap; gap: 6px; }
  .codex-toolbar .codex-file-chip {
    padding: 2px 6px; border: 1px solid #30363d; border-radius: 999px;
    background: #161b22; color: var(--fg); max-width: 280px;
  }
  .codex-toolbar .codex-file-chip input { margin: 0; }
  .codex-toolbar .codex-file-chip span {
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
  }
  .codex-toolbar .single-only[hidden], .codex-toolbar .stream-only[hidden] { display: none; }
  .codex-toolbar .meta { margin-left: auto; font-family: var(--mono); font-size: 11px; }
  .codex-toolbar .pid-alive { color: var(--good); }
  .codex-toolbar .pid-dead  { color: var(--muted); }
  .full { grid-column: 1 / -1; }
  canvas { max-height: 280px; }
  .err { color: var(--bad); font-family: var(--mono); }
  .status-dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%;
                margin-right: 6px; vertical-align: middle; }
  .status-dot.live { background: var(--good); animation: pulse 1.5s infinite; }
  .status-dot.idle { background: var(--muted); }
  .status-dot.dead { background: var(--bad); }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }
  table { width: 100%; border-collapse: collapse; font-family: var(--mono); font-size: 12px; }
  th, td { text-align: right; padding: 4px 8px; border-bottom: 1px solid var(--border); }
  th:first-child, td:first-child { text-align: left; }
  th { color: var(--muted); font-weight: 500; text-transform: uppercase;
       font-size: 11px; letter-spacing: 0.3px; }
</style>
</head>
<body>

<div class="topbar">
  <div>
    <h1><span class="status-dot" id="status-dot"></span>OSS Training Dashboard</h1>
    <div class="sub" id="header-sub">loading…</div>
  </div>
  <div class="run-picker">
    <label for="run-select">run</label>
    <select id="run-select"><option value="">loading…</option></select>
  </div>
</div>

<div class="grid">

  <div class="panel">
    <h2>Progress</h2>
    <div class="stat-row">
      <div class="stat">
        <div class="lbl">step</div>
        <div class="val" id="stat-step">–</div>
      </div>
      <div class="stat">
        <div class="lbl">elapsed</div>
        <div class="val" id="stat-elapsed">–</div>
      </div>
      <div class="stat">
        <div class="lbl">steps/sec</div>
        <div class="val" id="stat-rate">–</div>
      </div>
    </div>
    <div class="progress"><div class="progress-fill" id="progress-bar" style="width:0%"></div></div>
    <div class="sub" id="progress-text" style="margin-top:8px">–</div>
  </div>

  <div class="panel">
    <h2>Latest evaluation</h2>
    <div class="stat-row">
      <div class="stat">
        <div class="lbl">model PSNR</div>
        <div class="val" id="stat-model-psnr">–</div>
      </div>
      <div class="stat">
        <div class="lbl">bicubic PSNR</div>
        <div class="val" id="stat-bicubic-psnr">–</div>
      </div>
      <div class="stat">
        <div class="lbl">PSNR margin</div>
        <div class="val" id="stat-margin">–</div>
      </div>
      <div class="stat">
        <div class="lbl">PSNR beats / 8</div>
        <div class="val" id="stat-beats">–</div>
      </div>
    </div>
    <div class="stat-row">
      <div class="stat">
        <div class="lbl">model LPIPS ↓</div>
        <div class="val" id="stat-model-lpips">–</div>
      </div>
      <div class="stat">
        <div class="lbl">bicubic LPIPS ↓</div>
        <div class="val" id="stat-bicubic-lpips">–</div>
      </div>
      <div class="stat">
        <div class="lbl">LPIPS margin (↓ better)</div>
        <div class="val" id="stat-lpips-margin">–</div>
      </div>
      <div class="stat">
        <div class="lbl">LPIPS beats / 8</div>
        <div class="val" id="stat-lpips-beats">–</div>
      </div>
    </div>
    <div class="sub" id="eval-step-text">–</div>
  </div>

  <div class="panel full">
    <h2>In-flight comparison: <span style="color:#8b949e;font-weight:400;font-size:13px">LR-bilinear · bicubic · v4-baseline · v5-temporal · GT · |err| heatmap</span></h2>
    <div id="viz-meta" style="color:#8b949e;font-size:12px;margin-bottom:8px">loading…</div>
    <input type="range" id="viz-scrubber" min="0" max="0" value="0" style="width:100%;margin-bottom:8px" disabled>
    <div id="viz-zoom-host" style="overflow:auto;border:1px solid #30363d;border-radius:4px;max-height:80vh">
      <img id="viz-img" alt="(no viz yet)" style="display:block;cursor:zoom-in" data-zoomed="0">
    </div>
    <div style="color:#8b949e;font-size:11px;margin-top:4px">Click image to toggle 1:1 zoom (then drag to pan). Higher quality = sharper textures, fewer fringes; |err| heatmap: black = no error, red→yellow = increasing |model − GT|.</div>
  </div>

  <div class="panel">
    <div class="panel-head">
      <h2>PSNR <span style="font-size:13px;color:#3fb950">↑ higher is better</span></h2>
      <div class="chart-toolbar" data-chart="chart-psnr">
        <button data-action="zoom-out" title="Zoom out">−</button>
        <button data-action="zoom-in" title="Zoom in">+</button>
        <button data-action="reset" title="Reset zoom + pan">↺</button>
      </div>
    </div>
    <canvas id="chart-psnr"></canvas>
    <div class="chart-legend" id="legend-chart-psnr"></div>
    <div style="font-size:11px;color:#8b949e;margin-top:4px"><b>Two-finger scroll (or scroll-wheel) to pan · ⌘/Ctrl+scroll or pinch to zoom · double-click to reset.</b> Solid line: live training-time PSNR proxy (≈ −10·log10(t_l1²)). Held-out eval lines populate after `sr_temporal_held_out.py` runs (closeout). Dashed: published-benchmark estimates of competing upscalers at 1080p→4K Quality (±1 dB envelope). Solid + thicker = our measured numbers (OSS v3, v4).</div>
  </div>

  <div class="panel">
    <div class="panel-head">
      <h2>LPIPS-VGG <span style="font-size:13px;color:#3fb950">↓ lower is better</span></h2>
      <div class="chart-toolbar" data-chart="chart-lpips">
        <button data-action="zoom-out" title="Zoom out">−</button>
        <button data-action="zoom-in" title="Zoom in">+</button>
        <button data-action="reset" title="Reset zoom + pan">↺</button>
      </div>
    </div>
    <canvas id="chart-lpips"></canvas>
    <div class="chart-legend" id="legend-chart-lpips"></div>
    <div style="font-size:11px;color:#8b949e;margin-top:4px">Solid line: live training-time LPIPS (Phase 2+ only, when LPIPS loss is enabled). Held-out eval lines populate after closeout. Dashed: published-benchmark estimates (Bicubic ≈ 0.51, DLSS 2/FSR 2 ≈ 0.22, DLSS 4 ≈ 0.17).</div>
  </div>

  <div class="panel">
    <div class="panel-head">
      <h2>Throughput <span style="font-size:13px;color:#3fb950">↑ higher is better (steps/min)</span></h2>
      <div class="chart-toolbar" data-chart="chart-throughput">
        <button data-action="zoom-out" title="Zoom out">−</button>
        <button data-action="zoom-in" title="Zoom in">+</button>
        <button data-action="reset" title="Reset zoom + pan">↺</button>
      </div>
    </div>
    <canvas id="chart-throughput"></canvas>
    <div class="chart-legend" id="legend-chart-throughput"></div>
    <div style="font-size:11px;color:#8b949e;margin-top:4px">Computed from train-row timestamps. Sustained drop indicates DataLoader starvation or compute-bound phase (Phase 2 LPIPS-VGG cuts throughput ~5×).</div>
  </div>

  <div class="panel">
    <div class="panel-head">
      <h2>Loss decomposition <span style="font-size:13px;color:#3fb950">↓ lower is better</span></h2>
      <div class="chart-toolbar" data-chart="chart-loss">
        <button data-action="zoom-out" title="Zoom out">−</button>
        <button data-action="zoom-in" title="Zoom in">+</button>
        <button data-action="reset" title="Reset zoom + pan">↺</button>
      </div>
    </div>
    <canvas id="chart-loss"></canvas>
    <div class="chart-legend" id="legend-chart-loss"></div>
    <div style="font-size:11px;color:#8b949e;margin-top:4px">Phase 1: appearance loss (L1+SSIM) only. Phase 2 (step 10K+): adds LPIPS + temporal-consistency. Phase 3 (step 60K+): same loss, LR×0.01 polish.</div>
  </div>

  <div class="panel">
    <div class="panel-head">
      <h2>SSIM <span style="font-size:13px;color:#3fb950">↑ higher is better</span></h2>
      <div class="chart-toolbar" data-chart="chart-ssim">
        <button data-action="zoom-out" title="Zoom out">−</button>
        <button data-action="zoom-in" title="Zoom in">+</button>
        <button data-action="reset" title="Reset zoom + pan">↺</button>
      </div>
    </div>
    <canvas id="chart-ssim"></canvas>
    <div class="chart-legend" id="legend-chart-ssim"></div>
  </div>

  <div class="panel">
    <div class="panel-head">
      <h2>Output stats (mean, std)</h2>
      <div class="chart-toolbar" data-chart="chart-out">
        <button data-action="zoom-out" title="Zoom out">−</button>
        <button data-action="zoom-in" title="Zoom in">+</button>
        <button data-action="reset" title="Reset zoom + pan">↺</button>
      </div>
    </div>
    <canvas id="chart-out"></canvas>
    <div class="chart-legend" id="legend-chart-out"></div>
  </div>

  <div class="panel">
    <div class="panel-head">
      <h2>Gradient norms</h2>
      <div class="chart-toolbar" data-chart="chart-grad">
        <button data-action="zoom-out" title="Zoom out">−</button>
        <button data-action="zoom-in" title="Zoom in">+</button>
        <button data-action="reset" title="Reset zoom + pan">↺</button>
      </div>
    </div>
    <canvas id="chart-grad"></canvas>
    <div class="chart-legend" id="legend-chart-grad"></div>
  </div>

  <div class="panel full">
    <h2>Eval history</h2>
    <table id="eval-table">
      <thead>
        <tr><th>step</th><th>model</th><th>bicubic</th><th>margin</th><th>beats</th></tr>
      </thead>
      <tbody></tbody>
    </table>
  </div>

  <div class="panel full">
    <h2>Log tail</h2>
    <div class="log" id="log-tail">loading…</div>
  </div>

  <div class="panel full">
    <div class="panel-head">
      <h2>Codex live log</h2>
      <div class="codex-toolbar">
        <label><input type="checkbox" id="codex-stream-toggle" checked /> stream</label>
        <label class="stream-only"><input type="checkbox" id="codex-show-all-toggle" /> all logs (incl. >1h old)</label>
        <span class="stream-only codex-file-filters" id="codex-file-filters"></span>
        <label class="single-only" for="codex-file-select">file</label>
        <select class="single-only" id="codex-file-select"><option value="">(no codex logs found)</option></select>
        <label><input type="checkbox" id="codex-tail-toggle" checked /> auto-scroll</label>
        <label><input type="checkbox" id="codex-pause-toggle" /> pause</label>
        <span class="meta" id="codex-meta">–</span>
      </div>
    </div>
    <pre class="codex-log" id="codex-log">waiting for active codex logs…</pre>
  </div>

</div>

<script>
const POLL_MS = 10_000;
const STALE_MS = 60_000;
const RUN_STORAGE_KEY = 'oss-training-dashboard-run';

let charts = {};
let lastUpdate = null;
let currentRun = new URLSearchParams(window.location.search).get('run') || '';

const fmt = {
  num(x, d = 2) { return x === null || x === undefined || Number.isNaN(x) ? '–' : Number(x).toFixed(d); },
  sci(x) { return x === null || x === undefined || Number.isNaN(x) ? '–' : Number(x).toExponential(2); },
  duration(s) {
    if (s === null || s === undefined) return '–';
    s = Math.max(0, Math.floor(s));
    const h = Math.floor(s / 3600);
    const m = Math.floor((s % 3600) / 60);
    const ss = s % 60;
    if (h > 0) return `${h}h ${m}m ${ss}s`;
    if (m > 0) return `${m}m ${ss}s`;
    return `${ss}s`;
  },
};

function setStatus(state, text) {
  const dot = document.getElementById('status-dot');
  dot.className = 'status-dot ' + state;
  document.getElementById('header-sub').textContent = text;
}

function lineChart(canvasId, label, color, opts = {}) {
  const ctx = document.getElementById(canvasId).getContext('2d');
  return new Chart(ctx, {
    type: 'line',
    data: { datasets: [] },
    plugins: opts.extraPlugins || [],
    options: {
      animation: false,
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: 'nearest', axis: 'x', intersect: false },
      plugins: {
        // Built-in legend disabled: each chart panel renders its own
        // checkbox-style legend below the canvas (see installChartControls).
        legend: { display: false },
        // chartjs-plugin-zoom: pan via drag, zoom via Ctrl/Cmd-wheel or
        // pinch (trackpad). Plain scroll-wheel + two-finger swipe are
        // routed to chart.pan() by a custom canvas wheel handler in
        // installChartControls — that's the standard scroll-as-pan UX
        // most chart UIs ship with.
        zoom: {
          pan: { enabled: true, mode: 'xy', modifierKey: null },
          zoom: {
            wheel: { enabled: true, modifierKey: 'ctrl', speed: 0.1 },
            pinch: { enabled: true },
            mode: 'xy',
          },
          limits: {
            x: { min: 'original', max: 'original' },
            y: { min: 'original', max: 'original' },
          },
        },
      },
      scales: {
        x: { type: 'linear', title: { display: true, text: 'step', color: '#8b949e' },
             ticks: { color: '#8b949e' }, grid: { color: '#30363d' },
             // suggestedMin/Max: data-driven if present, falls back to these
             // bounds if the chart has no points yet (so reference lines + the
             // axis itself render at training-step scale instead of Chart.js's
             // default 0..1).
             ...(opts.xMin !== undefined ? { suggestedMin: opts.xMin } : {}),
             ...(opts.xMax !== undefined ? { suggestedMax: opts.xMax } : {}) },
        y: { title: { display: true, text: opts.yLabel || label, color: '#8b949e' },
             ticks: { color: '#8b949e' }, grid: { color: '#30363d' },
             ...(opts.yMin !== undefined ? { min: opts.yMin } : {}),
             ...(opts.yMax !== undefined ? { max: opts.yMax } : {}) },
      },
    },
  });
}

function setChart(chart, datasets) {
  // Preserve user's per-dataset visibility selections across data refresh.
  const prevHidden = {};
  for (const ds of (chart.data.datasets || [])) {
    if (ds && ds.label != null) prevHidden[ds.label] = !!ds.hidden;
  }
  for (const ds of datasets) {
    if (ds && ds.label != null && ds.label in prevHidden) {
      ds.hidden = prevHidden[ds.label];
    }
  }
  chart.data.datasets = datasets;
  chart.update();
  refreshChartLegend(chart);
}

function _clearChildren(node) {
  while (node.firstChild) node.removeChild(node.firstChild);
}

function refreshChartLegend(chart) {
  const id = chart.canvas.id;
  const container = document.getElementById('legend-' + id);
  if (!container) return;
  _clearChildren(container);
  const datasets = chart.data.datasets || [];
  if (datasets.length === 0) {
    container.textContent = '(no data yet)';
    container.style.color = 'var(--muted)';
    return;
  }
  container.style.color = '';
  for (let i = 0; i < datasets.length; i++) {
    const ds = datasets[i];
    if (!ds || ds.label == null) continue;
    const label = document.createElement('label');
    if (ds.hidden) label.classList.add('off');
    const cb = document.createElement('input');
    cb.type = 'checkbox';
    cb.checked = !ds.hidden;
    cb.addEventListener('change', () => {
      ds.hidden = !cb.checked;
      if (ds.hidden) label.classList.add('off');
      else label.classList.remove('off');
      chart.update();
    });
    const swatch = document.createElement('span');
    swatch.className = 'swatch';
    swatch.style.background = ds.borderColor || ds.backgroundColor || '#888';
    label.appendChild(cb);
    label.appendChild(swatch);
    label.appendChild(document.createTextNode(' ' + ds.label));
    container.appendChild(label);
  }
}

function installChartControls(chart) {
  const id = chart.canvas.id;
  const toolbar = document.querySelector('.chart-toolbar[data-chart="' + id + '"]');
  if (toolbar) {
    toolbar.addEventListener('click', (ev) => {
      const btn = ev.target.closest('button');
      if (!btn) return;
      const action = btn.getAttribute('data-action');
      if (action === 'zoom-in') chart.zoom(1.25);
      else if (action === 'zoom-out') chart.zoom(0.8);
      else if (action === 'reset') chart.resetZoom();
    });
  }
  // Plain wheel/two-finger scroll = pan; Ctrl/Cmd+wheel = zoom (delegated
  // to chartjs-plugin-zoom which has wheel.modifierKey='ctrl').
  const canvas = chart.canvas;
  canvas.addEventListener('wheel', (ev) => {
    if (ev.ctrlKey || ev.metaKey) return;
    ev.preventDefault();
    const dx = -ev.deltaX;
    const dy = -ev.deltaY;
    if (chart.pan && (dx || dy)) {
      chart.pan({ x: dx, y: dy }, undefined, 'default');
    }
  }, { passive: false });
  refreshChartLegend(chart);
}

// Published-benchmark estimates of competing real-time upscalers at
// 1080p->4K Quality mode on RTX <train-host>-class hardware. Values are
// approximate (±1 dB on PSNR, ±0.05 on LPIPS, ±0.5 ms on latency); they
// vary significantly by scene content and were never directly measured
// by us. Sources: NVIDIA / AMD whitepapers + independent benchmark
// roundups (e.g. ComputerBase, TechPowerUp). DLSS 5 omitted: not
// publicly released as of this dashboard's authoring.
const UPSCALER_ESTIMATES = {
  // [psnr_dB, lpips, latency_ms, color]
  bicubic:  { psnr: 25.8, lpips: 0.51, latency_ms: 0.05, color: '#8b949e' },
  fsr1:     { psnr: 26.5, lpips: 0.45, latency_ms: 0.4,  color: '#a371f7' },
  fsr2:     { psnr: 28.5, lpips: 0.28, latency_ms: 0.8,  color: '#bc8cff' },
  fsr3:     { psnr: 28.5, lpips: 0.28, latency_ms: 0.8,  color: '#d2a8ff' },
  fsr4:     { psnr: 30.0, lpips: 0.22, latency_ms: 2.0,  color: '#e6c1ff' },
  dlss1:    { psnr: 26.5, lpips: 0.40, latency_ms: 1.5,  color: '#1f6feb' },
  dlss2:    { psnr: 30.0, lpips: 0.22, latency_ms: 0.4,  color: '#3fb950' },
  dlss3:    { psnr: 30.0, lpips: 0.22, latency_ms: 0.4,  color: '#56d364' },
  dlss4:    { psnr: 31.5, lpips: 0.17, latency_ms: 1.0,  color: '#7ee787' },
  // OSS measured baselines (NOT estimates — actual held-out numbers from v3-vs-v4 A/B
  // memo on CitySample, n=64, fixed-batch, shuffle=False, manual_seed=0).
  oss_v3:   { psnr: 29.6, lpips: 0.40, latency_ms: 37.6, color: '#f0883e', kind: 'ours' },
  oss_v4:   { psnr: 29.5, lpips: 0.31, latency_ms: 37.6, color: '#ff9b3c', kind: 'ours' },
};

// Phase markers — vertical lines on every step-axis chart at the
// Phase 1 -> Phase 2 (step 10000) and Phase 2 -> Phase 3 (step 60000)
// transitions. Drawn via a tiny Chart.js plugin.
const PHASE_MARKERS = [
  { step: 10000, label: 'Phase 1→2', color: '#d29922' },
  { step: 60000, label: 'Phase 2→3', color: '#3fb950' },
];

const phaseMarkerPlugin = {
  id: 'phaseMarkers',
  afterDraw(chart) {
    const x = chart.scales.x; if (!x) return;
    const y = chart.scales.y; if (!y) return;
    const ctx = chart.ctx;
    ctx.save();
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 3]);
    ctx.font = '10px -apple-system, sans-serif';
    for (const m of PHASE_MARKERS) {
      if (m.step < x.min || m.step > x.max) continue;
      const xp = x.getPixelForValue(m.step);
      ctx.strokeStyle = m.color;
      ctx.beginPath();
      ctx.moveTo(xp, y.top);
      ctx.lineTo(xp, y.bottom);
      ctx.stroke();
      ctx.fillStyle = m.color;
      ctx.fillText(m.label, xp + 4, y.top + 12);
    }
    ctx.restore();
  },
};

const upscalerRefPlugin = (metric) => ({
  id: 'upscalerRef-' + metric,
  afterDatasetsDraw(chart) {
    const x = chart.scales.x; if (!x) return;
    const y = chart.scales.y; if (!y) return;
    const ctx = chart.ctx;
    ctx.save();
    ctx.setLineDash([3, 4]);
    ctx.lineWidth = 1;
    ctx.font = '10px -apple-system, sans-serif';
    const items = [
      ['bicubic', 'bicubic'], ['fsr1', 'FSR 1'], ['fsr2', 'FSR 2'],
      ['fsr3', 'FSR 3'], ['fsr4', 'FSR 4'], ['dlss1', 'DLSS 1'],
      ['dlss2', 'DLSS 2'], ['dlss3', 'DLSS 3'], ['dlss4', 'DLSS 4'],
      ['oss_v3', 'OSS v3 (ours, measured)'], ['oss_v4', 'OSS v4 (ours, measured)'],
    ];

    // First pass: draw all reference lines and collect label-placement
    // candidates. Lines stay at their true y; only labels get adjusted to
    // avoid overlap.
    const candidates = [];
    for (const [k, label] of items) {
      const e = UPSCALER_ESTIMATES[k];
      const v = e[metric];
      if (v == null) continue;
      const yp = y.getPixelForValue(v);
      if (yp < y.top || yp > y.bottom) continue;
      ctx.strokeStyle = e.color;
      if (e.kind === 'ours') {
        ctx.setLineDash([]);
        ctx.lineWidth = 2;
      } else {
        ctx.setLineDash([3, 4]);
        ctx.lineWidth = 1;
      }
      ctx.beginPath();
      ctx.moveTo(x.left, yp);
      ctx.lineTo(x.right, yp);
      ctx.stroke();
      const suffix = e.kind === 'ours' ? '' : ' (est)';
      candidates.push({ y: yp, color: e.color, text: label + suffix, kind: e.kind });
    }

    // Second pass: vertically de-overlap label y-positions while keeping
    // the line y-positions intact. Sort by y ascending, walk top->bottom,
    // and if a candidate would overlap the previous label, push it down by
    // a minimum spacing. Two-pass: forward (for collisions on the way down)
    // then backward (for any candidates that ended up below the chart).
    const MIN_SPACING = 12;  // 10px font + 2px padding
    candidates.sort((a, b) => a.y - b.y);
    for (let i = 1; i < candidates.length; i++) {
      const prev = candidates[i - 1];
      if (candidates[i].y - prev.y < MIN_SPACING) {
        candidates[i].y = prev.y + MIN_SPACING;
      }
    }
    for (let i = candidates.length - 2; i >= 0; i--) {
      const next = candidates[i + 1];
      if (next.y > y.bottom - 4 && next.y - candidates[i].y < MIN_SPACING) {
        candidates[i].y = next.y - MIN_SPACING;
        next.y = y.bottom - 4;
      }
    }

    // Draw labels at their adjusted positions. If a label was nudged away
    // from its reference line, draw a thin 1px leader from line-y to
    // label-y so the user can still see which line each label refers to.
    ctx.setLineDash([]);
    for (const c of candidates) {
      const trueY = y.getPixelForValue(
        UPSCALER_ESTIMATES[Object.keys(UPSCALER_ESTIMATES).find(k => {
          const e = UPSCALER_ESTIMATES[k];
          const lbl = items.find(it => it[0] === k);
          if (!lbl) return false;
          const suffix = e.kind === 'ours' ? '' : ' (est)';
          return (lbl[1] + suffix) === c.text;
        })][metric]
      );
      if (Math.abs(trueY - c.y) > 2) {
        ctx.strokeStyle = c.color;
        ctx.globalAlpha = 0.5;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(x.right - 6, trueY);
        ctx.lineTo(x.right - 4, c.y);
        ctx.stroke();
        ctx.globalAlpha = 1;
      }
      ctx.fillStyle = c.color;
      ctx.fillText(c.text, x.right - 110, c.y - 2);
    }
    ctx.restore();
  },
});

// Double-click any chart to reset its zoom/pan to original bounds.
function _wireResetZoomOnDblClick() {
  const ids = ['chart-psnr', 'chart-lpips', 'chart-throughput', 'chart-loss',
               'chart-ssim', 'chart-out', 'chart-grad'];
  for (const id of ids) {
    const el = document.getElementById(id);
    if (!el) continue;
    el.addEventListener('dblclick', () => {
      const c = charts[id.replace('chart-', '')];
      if (c && typeof c.resetZoom === 'function') c.resetZoom();
    });
  }
}

function buildCharts() {
  // Explicit y-min/y-max so the upscaler reference lines render even when
  // there's no data yet (Chart.js can't compute axis bounds from 0 points,
  // and getPixelForValue returns NaN without bounds).
  // Default x-axis bound matches the v5 / v6 training horizon (80K steps).
  // Used as suggestedMax so live data can extend it but the empty-chart
  // case still renders at training-step scale instead of Chart.js's 0..1
  // default.
  const X_MAX_DEFAULT = 80000;
  charts.psnr = lineChart('chart-psnr', 'PSNR (dB)', null, {
    yLabel: 'PSNR (dB) ↑ better', yMin: 24, yMax: 36,
    xMin: 0, xMax: X_MAX_DEFAULT,
    extraPlugins: [phaseMarkerPlugin, upscalerRefPlugin('psnr')],
  });
  charts.lpips = lineChart('chart-lpips', 'LPIPS', null, {
    yLabel: 'LPIPS ↓ better', yMin: 0, yMax: 0.55,
    xMin: 0, xMax: X_MAX_DEFAULT,
    extraPlugins: [phaseMarkerPlugin, upscalerRefPlugin('lpips')],
  });
  charts.throughput = lineChart('chart-throughput', 'steps/min', null, {
    yLabel: 'steps/min ↑ better', yMin: 0,
    xMin: 0, xMax: X_MAX_DEFAULT,
    extraPlugins: [phaseMarkerPlugin],
  });
  charts.loss = lineChart('chart-loss', 'loss', null, {
    yLabel: 'loss ↓ better',
    xMin: 0, xMax: X_MAX_DEFAULT,
    extraPlugins: [phaseMarkerPlugin],
  });
  charts.ssim = lineChart('chart-ssim', 'SSIM', null, {
    yLabel: 'SSIM ↑ better', yMin: 0, yMax: 1,
    xMin: 0, xMax: X_MAX_DEFAULT,
    extraPlugins: [phaseMarkerPlugin],
  });
  charts.out = lineChart('chart-out', 'output', null, {
    xMin: 0, xMax: X_MAX_DEFAULT,
  });
  charts.grad = lineChart('chart-grad', 'grad norm', null, {
    xMin: 0, xMax: X_MAX_DEFAULT,
  });
  // Wire toolbar buttons + custom checkbox legend + scroll-as-pan handler
  // for every chart we just built.
  for (const chart of Object.values(charts)) {
    installChartControls(chart);
  }
}

async function fetchJSON(url) {
  const r = await fetch(url, { cache: 'no-store' });
  if (!r.ok) throw new Error(`${url}: ${r.status}`);
  return r.json();
}

async function fetchText(url) {
  const r = await fetch(url, { cache: 'no-store' });
  if (!r.ok) throw new Error(`${url}: ${r.status}`);
  return r.text();
}

function withRun(url) {
  const u = new URL(url, window.location.origin);
  if (currentRun) u.searchParams.set('run', currentRun);
  return u.pathname + u.search;
}

async function initRunPicker() {
  const select = document.getElementById('run-select');
  if (!select) return;
  try {
    const payload = await fetchJSON('/api/runs');
    const runs = (payload && payload.runs) || [];
    const names = new Set(runs.map(r => r.name));
    const queryRun = new URLSearchParams(window.location.search).get('run') || '';
    const storedRun = localStorage.getItem(RUN_STORAGE_KEY) || '';
    if (queryRun && names.has(queryRun)) {
      currentRun = queryRun;
    } else if (!queryRun && storedRun && names.has(storedRun)) {
      currentRun = storedRun;
    } else {
      currentRun = payload.default_run || (runs[0] && runs[0].name) || '';
    }
    if (currentRun) localStorage.setItem(RUN_STORAGE_KEY, currentRun);

    while (select.firstChild) select.removeChild(select.firstChild);
    if (!runs.length) {
      const opt = document.createElement('option');
      opt.value = '';
      opt.textContent = '(no matching runs)';
      select.appendChild(opt);
      select.disabled = true;
      return;
    }
    select.disabled = false;
    for (const run of runs) {
      const opt = document.createElement('option');
      opt.value = run.name;
      const bits = [];
      if (run.has_train_log) bits.push('metrics');
      if (run.has_score_log) bits.push('score');
      if (run.has_viz) bits.push('viz');
      opt.textContent = bits.length ? `${run.name} (${bits.join(', ')})` : run.name;
      select.appendChild(opt);
    }
    select.value = currentRun;
    select.addEventListener('change', () => {
      const run = select.value;
      localStorage.setItem(RUN_STORAGE_KEY, run);
      const next = new URL(window.location.href);
      next.searchParams.set('run', run);
      window.location.href = next.toString();
    });
  } catch (e) {
    while (select.firstChild) select.removeChild(select.firstChild);
    const opt = document.createElement('option');
    opt.value = '';
    opt.textContent = '(run scan failed)';
    select.appendChild(opt);
    select.disabled = true;
  }
}

function renderEvalTable(scoreRows) {
  const tbody = document.querySelector('#eval-table tbody');
  // Clear existing rows safely.
  while (tbody.firstChild) tbody.removeChild(tbody.firstChild);

  const recent = scoreRows.slice(-20).reverse();
  for (const r of recent) {
    const margin = (r.model_psnr_mean ?? 0) - (r.bicubic_psnr_mean ?? 0);
    const tr = document.createElement('tr');

    const tdStep = document.createElement('td');
    tdStep.textContent = (r.step ?? 0).toLocaleString();
    tr.appendChild(tdStep);

    const tdModel = document.createElement('td');
    tdModel.textContent = fmt.num(r.model_psnr_mean);
    tr.appendChild(tdModel);

    const tdBicubic = document.createElement('td');
    tdBicubic.textContent = fmt.num(r.bicubic_psnr_mean);
    tr.appendChild(tdBicubic);

    const tdMargin = document.createElement('td');
    tdMargin.textContent = (margin >= 0 ? '+' : '') + fmt.num(margin);
    tdMargin.style.color = margin > 0.5 ? '#3fb950' : margin > -0.1 ? '#d29922' : '#f85149';
    tr.appendChild(tdMargin);

    const tdBeats = document.createElement('td');
    tdBeats.textContent = String(r.model_beats_bicubic_count ?? '–');
    tr.appendChild(tdBeats);

    tbody.appendChild(tr);
  }
}

// Viz scrubber: list step-XXXXX.png files, latest first; slider scrubs the
// timeline, defaults to the most recent. ``follow`` is true when the slider
// is at max (so new ckpts auto-advance the image).
let vizFiles = [];
let vizFollow = true;

function _fileToStep(fname) {
  const m = fname.match(/step-(\d+)\.png/);
  return m ? parseInt(m[1]) : -1;
}

function refreshViz(files) {
  files = (files || []).slice().sort((a, b) => _fileToStep(a) - _fileToStep(b));
  vizFiles = files;
  const meta = document.getElementById('viz-meta');
  const img = document.getElementById('viz-img');
  const slider = document.getElementById('viz-scrubber');
  if (!files.length) {
    meta.textContent = 'no PNGs yet · viz loop renders one ~5 min after first ckpt';
    img.removeAttribute('src');
    img.alt = '(no viz yet)';
    slider.disabled = true;
    slider.max = 0;
    return;
  }
  slider.disabled = false;
  slider.max = files.length - 1;
  if (vizFollow) {
    slider.value = files.length - 1;
  }
  const idx = Math.min(parseInt(slider.value), files.length - 1);
  const fname = files[idx];
  const step = _fileToStep(fname);
  maybeFlashTitle(step);
  img.src = withRun('/viz/' + fname + '?_t=' + Date.now());
  img.alt = 'step ' + step;
  meta.textContent = 'step ' + step.toLocaleString() +
                     ' · ' + files.length + ' ckpt(s) rendered · ' +
                     (vizFollow ? 'following latest (drag slider to pin)' : 'pinned (move to right edge to follow)');
}

document.addEventListener('DOMContentLoaded', () => {
  const slider = document.getElementById('viz-scrubber');
  if (slider) {
    slider.addEventListener('input', () => {
      vizFollow = (parseInt(slider.value) === parseInt(slider.max));
      refreshViz(vizFiles);
    });
  }
  // Click-to-zoom: toggle 1:1 image size so user can scroll-pan into a
  // region. Default fit-to-width via parent overflow:auto + max-width 100%.
  const img = document.getElementById('viz-img');
  const host = document.getElementById('viz-zoom-host');
  if (img && host) {
    img.style.maxWidth = '100%';
    img.addEventListener('click', () => {
      const zoomed = img.dataset.zoomed === '1';
      if (zoomed) {
        img.style.maxWidth = '100%';
        img.style.width = 'auto';
        img.style.cursor = 'zoom-in';
        img.dataset.zoomed = '0';
      } else {
        img.style.maxWidth = 'none';
        img.style.width = img.naturalWidth + 'px';
        img.style.cursor = 'zoom-out';
        img.dataset.zoomed = '1';
      }
    });
  }
});

// Tab title flash: when a new step PNG arrives while the tab is in
// background, change the page title so the user notices the update.
let _vizPrevStep = -1;
function maybeFlashTitle(latestStep) {
  if (latestStep > _vizPrevStep && _vizPrevStep !== -1 && document.hidden) {
    document.title = '🔴 step ' + latestStep.toLocaleString() + ' — OSS dashboard';
  } else if (!document.hidden) {
    document.title = 'OSS Training Dashboard';
  }
  _vizPrevStep = Math.max(_vizPrevStep, latestStep);
}
document.addEventListener('visibilitychange', () => {
  if (!document.hidden) document.title = 'OSS Training Dashboard';
});

async function refresh() {
  try {
    const [info, metrics, score, logTail, viz] = await Promise.all([
      fetchJSON(withRun('/api/info')),
      fetchJSON(withRun('/api/metrics')),
      fetchJSON(withRun('/api/score')),
      fetchText(withRun('/api/log')),
      fetchJSON(withRun('/api/viz')).catch(() => ({ files: [] })),
    ]);

    lastUpdate = Date.now();
    refreshViz(viz && viz.files ? viz.files : []);

    const train = (metrics && metrics.train) || metrics || [];
    const scoreRows = score || [];

    // ---- Progress card ----
    const lastTrain = train[train.length - 1];
    const step = lastTrain ? lastTrain.step : 0;
    const maxSteps = info.max_steps || 0;
    const pct = maxSteps > 0 ? Math.min(100, (step / maxSteps) * 100) : 0;
    document.getElementById('stat-step').textContent =
      step.toLocaleString() + ' / ' + (maxSteps ? maxSteps.toLocaleString() : '?');
    document.getElementById('progress-bar').style.width = pct.toFixed(1) + '%';

    let elapsed = info.elapsed_seconds;
    document.getElementById('stat-elapsed').textContent = fmt.duration(elapsed);

    let rate = null;
    if (info.start_time_unix && info.last_log_time_unix && step > 0) {
      const dt = info.last_log_time_unix - info.start_time_unix;
      if (dt > 0) rate = step / dt;
    }
    document.getElementById('stat-rate').textContent = rate ? fmt.num(rate, 1) : '–';

    const wallBudget = info.max_time_seconds;
    let progressText = '';
    if (wallBudget && elapsed !== null) {
      const remaining = Math.max(0, wallBudget - elapsed);
      progressText = `wall budget ${fmt.duration(wallBudget)} · remaining ${fmt.duration(remaining)}`;
    } else if (maxSteps && step > 0) {
      progressText = `${pct.toFixed(1)}% of max_steps`;
    }
    document.getElementById('progress-text').textContent = progressText;

    // ---- Latest eval card ----
    const lastEval = scoreRows[scoreRows.length - 1];
    if (lastEval) {
      const m = lastEval.model_psnr_mean;
      const b = lastEval.bicubic_psnr_mean;
      const beats = lastEval.model_beats_bicubic_count;
      const margin = m - b;
      document.getElementById('stat-model-psnr').textContent = fmt.num(m) + ' dB';
      document.getElementById('stat-bicubic-psnr').textContent = fmt.num(b) + ' dB';
      const marginEl = document.getElementById('stat-margin');
      marginEl.textContent = (margin >= 0 ? '+' : '') + fmt.num(margin) + ' dB';
      marginEl.className = 'val ' + (margin > 0.5 ? 'good' : margin > -0.1 ? 'warn' : 'bad');
      document.getElementById('stat-beats').textContent = String(beats);

      // LPIPS card (lower=better; 'beats' means model LPIPS < bicubic LPIPS).
      if (lastEval.model_lpips_mean !== undefined && lastEval.model_lpips_mean !== null) {
        const lm = lastEval.model_lpips_mean;
        const lb = lastEval.bicubic_lpips_mean;
        const lbeats = lastEval.model_beats_bicubic_lpips_count;
        const lmargin = lb - lm; // positive = model better
        document.getElementById('stat-model-lpips').textContent = fmt.num(lm, 4);
        document.getElementById('stat-bicubic-lpips').textContent = fmt.num(lb, 4);
        const lmEl = document.getElementById('stat-lpips-margin');
        lmEl.textContent = (lmargin >= 0 ? '−' : '+') + fmt.num(Math.abs(lmargin), 4);
        lmEl.className = 'val ' + (lmargin > 0.005 ? 'good' : lmargin > -0.001 ? 'warn' : 'bad');
        document.getElementById('stat-lpips-beats').textContent = String(lbeats);
      } else {
        document.getElementById('stat-model-lpips').textContent = '–';
        document.getElementById('stat-bicubic-lpips').textContent = '–';
        document.getElementById('stat-lpips-margin').textContent = '–';
        document.getElementById('stat-lpips-beats').textContent = '–';
      }

      document.getElementById('eval-step-text').textContent =
        'evaluated at step ' + lastEval.step.toLocaleString();
    } else {
      document.getElementById('eval-step-text').textContent =
        'no eval yet — first one at step ' + (info.eval_every || 5000).toLocaleString();
    }

    // ---- Eval history table (safe DOM build) ----
    renderEvalTable(scoreRows);

    // ---- Charts ----
    // Fallback: if a row doesn't have ``key`` but has alt aliases, use the
    // first alias that exists. This handles v5-pixel-temporal rows whose
    // keys are ``t_l1`` / ``t_ssim`` instead of ``l1`` / ``ssim``.
    const trainXY = (key, ...aliases) => train.map(r => {
      let v = r[key];
      if ((v === undefined || v === null) && aliases.length) {
        for (const a of aliases) {
          if (r[a] !== undefined && r[a] !== null) { v = r[a]; break; }
        }
      }
      return { x: r.step, y: v };
    }).filter(p => p.y !== undefined && p.y !== null);

    setChart(charts.loss, [
      { label: 'total loss', data: trainXY('loss'), borderColor: '#58a6ff', backgroundColor: '#58a6ff', tension: 0, pointRadius: 0 },
      { label: 't_l1', data: trainXY('l1', 't_l1'), borderColor: '#d29922', backgroundColor: '#d29922', tension: 0, pointRadius: 0 },
      { label: 'tp1_l1', data: trainXY('tp1_l1'), borderColor: '#a371f7', backgroundColor: '#a371f7', tension: 0, pointRadius: 0 },
      { label: 't_lpips', data: trainXY('t_lpips'), borderColor: '#f85149', backgroundColor: '#f85149', tension: 0, pointRadius: 0 },
      { label: 'tp1_lpips', data: trainXY('tp1_lpips'), borderColor: '#ff7b72', backgroundColor: '#ff7b72', tension: 0, pointRadius: 0 },
      { label: 'tc (temporal-consistency)', data: trainXY('tc'), borderColor: '#3fb950', backgroundColor: '#3fb950', tension: 0, pointRadius: 0 },
    ]);

    // Throughput series: rolling steps/min computed from train-row timestamps.
    // Each row carries a ``step`` and we receive an info.last_log_time_unix +
    // start_time_unix; for inter-step throughput we approximate via the
    // delta of step-vs-step assuming the polling interval is ~constant. A
    // more accurate version would store a per-row timestamp; this is an
    // intent-correct proxy for now.
    const throughputXY = [];
    if (train.length >= 2 && info && info.start_time_unix && info.last_log_time_unix) {
      const totalSteps = train[train.length - 1].step - train[0].step;
      const totalSec = info.last_log_time_unix - info.start_time_unix;
      const stepsPerMin = totalSec > 0 ? (totalSteps / totalSec) * 60 : 0;
      // Plot constant series so user sees a recent average; rolling-window
      // would require per-row timestamps we don't currently emit.
      for (let i = 0; i < train.length; i += Math.max(1, Math.floor(train.length / 80))) {
        throughputXY.push({ x: train[i].step, y: stepsPerMin });
      }
    }
    setChart(charts.throughput, [
      { label: 'steps/min (run-avg)', data: throughputXY, borderColor: '#58a6ff', backgroundColor: '#58a6ff', tension: 0, pointRadius: 0 },
    ]);

    setChart(charts.ssim, [
      { label: 'SSIM (or t_ssim)', data: trainXY('ssim', 't_ssim'), borderColor: '#3fb950', backgroundColor: '#3fb950', tension: 0, pointRadius: 0 },
      { label: 'tp1_ssim', data: trainXY('tp1_ssim'), borderColor: '#a371f7', backgroundColor: '#a371f7', tension: 0, pointRadius: 0 },
    ]);

    setChart(charts.out, [
      { label: 'out_mean', data: trainXY('sr_out_mean'), borderColor: '#58a6ff', tension: 0, pointRadius: 0 },
      { label: 'out_std',  data: trainXY('sr_out_std'),  borderColor: '#d29922', tension: 0, pointRadius: 0 },
    ]);

    setChart(charts.grad, [
      { label: 'head conv', data: trainXY('sr_head_conv_grad_norm'), borderColor: '#58a6ff', tension: 0, pointRadius: 0 },
      { label: 'upsample conv', data: trainXY('sr_upsample_conv_grad_norm'), borderColor: '#d29922', tension: 0, pointRadius: 0 },
    ]);

    const evalXY = (key) => scoreRows
      .map(r => ({ x: r.step, y: r[key] }))
      .filter(p => p.y !== undefined && p.y !== null);
    // Live training-time PSNR proxy (rough: −10·log10(t_l1²) when t_l1 ≈ sqrt(MSE)).
    // Captures the relative trend; absolute values may differ from held-out PSNR
    // by 1-2 dB. Refresh once held-out eval rows arrive in score_log.json.
    const trainPsnrXY = train.map(r => {
      const v = (r.l1 != null) ? r.l1 : r.t_l1;
      if (v == null || v <= 0) return null;
      return { x: r.step, y: -10 * Math.log10(v * v) };
    }).filter(p => p !== null);
    const trainLpipsXY = trainXY('t_lpips');

    setChart(charts.psnr, [
      { label: 'live train PSNR proxy', data: trainPsnrXY, borderColor: '#58a6ff', backgroundColor: '#58a6ff', tension: 0, pointRadius: 0 },
      { label: 'held-out model (after closeout)', data: evalXY('model_psnr_mean'), borderColor: '#3fb950', backgroundColor: '#3fb950', tension: 0, pointRadius: 3 },
      { label: 'bicubic', data: evalXY('bicubic_psnr_mean'), borderColor: '#8b949e', backgroundColor: '#8b949e', tension: 0, pointRadius: 3, borderDash: [4, 4] },
    ]);
    setChart(charts.lpips, [
      { label: 'live train LPIPS (Phase 2+)', data: trainLpipsXY, borderColor: '#58a6ff', backgroundColor: '#58a6ff', tension: 0, pointRadius: 0 },
      { label: 'held-out model (after closeout)', data: evalXY('model_lpips_mean'), borderColor: '#3fb950', backgroundColor: '#3fb950', tension: 0, pointRadius: 3 },
      { label: 'held-out bicubic (after closeout)', data: evalXY('bicubic_lpips_mean'), borderColor: '#8b949e', backgroundColor: '#8b949e', tension: 0, pointRadius: 3, borderDash: [4, 4] },
    ]);

    // ---- Log tail (textContent — safe) ----
    document.getElementById('log-tail').textContent = logTail || '(empty)';
    document.getElementById('log-tail').scrollTop = 999999;

    // ---- Status ----
    const ageMs = Date.now() - (info.last_log_mtime_unix || 0) * 1000;
    if (info.last_log_mtime_unix && ageMs < STALE_MS) {
      setStatus('live', `output_dir: ${info.output_dir} · log: ${info.log_file} · updated ${Math.floor(ageMs/1000)}s ago`);
    } else if (info.last_log_mtime_unix) {
      setStatus('idle', `output_dir: ${info.output_dir} · log idle for ${Math.floor(ageMs/1000)}s — process may have stopped`);
    } else {
      setStatus('dead', `output_dir: ${info.output_dir} · no log activity`);
    }
  } catch (e) {
    setStatus('dead', 'fetch error: ' + e.message);
  }
}

buildCharts();
_wireResetZoomOnDblClick();
initRunPicker().finally(refresh);
setInterval(refresh, POLL_MS);

// ============================================================
// Codex live log panel — independent poller, faster cadence.
// ============================================================
const CODEX_POLL_MS = 2500;
const CODEX_FILE_KEY = 'oss-training-dashboard-codex-file';
const CODEX_MODE_KEY = 'oss-training-dashboard-codex-mode';
const CODEX_DISABLED_KEY = 'oss-training-dashboard-codex-disabled-files';
let codexCurrentFile = localStorage.getItem(CODEX_FILE_KEY) || '';
let codexStreamMode = localStorage.getItem(CODEX_MODE_KEY) !== 'single';
let codexFiles = [];

function escapeHTML(s) {
  return String(s).replace(/[&<>"']/g, c => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
  }[c]));
}

function codexDisabledFiles() {
  try { return new Set(JSON.parse(localStorage.getItem(CODEX_DISABLED_KEY) || '[]')); }
  catch (e) { return new Set(); }
}

function saveCodexDisabledFiles(disabled) {
  localStorage.setItem(CODEX_DISABLED_KEY, JSON.stringify(Array.from(disabled).sort()));
}

function setCodexModeUI() {
  const toggle = document.getElementById('codex-stream-toggle');
  if (toggle) toggle.checked = codexStreamMode;
  const showAll = document.getElementById('codex-show-all-toggle');
  if (showAll) showAll.checked = codexShowAll;
  document.querySelectorAll('.single-only').forEach(el => { el.hidden = codexStreamMode; });
  document.querySelectorAll('.stream-only').forEach(el => { el.hidden = !codexStreamMode; });
}

// Stream mode default: show ALL logs, with the active subset prioritized
// in the UI. The previous "active-only" filter hid every log older than
// 60 minutes, which broke the use case of scrolling through historical
// codex activity. The 'codex-show-all-toggle' switches between
// active-only and full history; default = full history.
let codexShowAll = localStorage.getItem('oss-training-dashboard-codex-show-all') !== 'false';

function streamCodexFiles() {
  return codexShowAll ? codexFiles : codexFiles.filter(f => f.active);
}

function activeCodexFiles() {
  // Backward-compat alias used elsewhere in the panel.
  return streamCodexFiles();
}

function selectedCodexStreamFiles() {
  const disabled = codexDisabledFiles();
  return streamCodexFiles().filter(f => !disabled.has(f.name)).map(f => f.name);
}

async function refreshCodexFileList() {
  try {
    const data = await fetchJSON('/api/codex-logs');
    const files = (data && data.files) || [];
    codexFiles = files;
    const sel = document.getElementById('codex-file-select');
    if (sel) {
      const prev = sel.value || codexCurrentFile;
      if (files.length === 0) {
        sel.innerHTML = '<option value="">(no codex logs found)</option>';
        sel.value = '';
        codexCurrentFile = '';
      } else {
        sel.innerHTML = files.map(f => {
          const tag = f.alive ? '● live' : (f.active ? '◐ active' : '○ old');
          const ageS = Math.max(0, Math.floor((Date.now() / 1000) - f.mtime));
          const kb = (f.size / 1024).toFixed(0);
          return `<option value="${escapeHTML(f.name)}">${tag} · ${escapeHTML(f.name)} · ${kb}KB · ${ageS}s</option>`;
        }).join('');
        if (prev && files.some(f => f.name === prev)) {
          sel.value = prev;
          codexCurrentFile = prev;
        } else {
          const live = files.find(f => f.alive);
          const active = files.find(f => f.active);
          const pick = (live || active || files[0]).name;
          sel.value = pick;
          codexCurrentFile = pick;
          localStorage.setItem(CODEX_FILE_KEY, pick);
        }
      }
    }
    const filters = document.getElementById('codex-file-filters');
    if (filters) {
      const disabled = codexDisabledFiles();
      const active = activeCodexFiles();
      filters.innerHTML = active.length === 0
        ? '<span class="codex-file-chip">no active logs</span>'
        : active.map(f => {
          const checked = disabled.has(f.name) ? '' : ' checked';
          const kb = (f.size / 1024).toFixed(0);
          return `<label class="codex-file-chip" title="${escapeHTML(f.name)}"><input type="checkbox" data-codex-file="${escapeHTML(f.name)}"${checked} /><span>${escapeHTML(f.name)} · ${kb}KB</span></label>`;
        }).join('');
      filters.querySelectorAll('input[data-codex-file]').forEach(input => {
        input.addEventListener('change', () => {
          const nextDisabled = codexDisabledFiles();
          const name = input.getAttribute('data-codex-file');
          if (!name) return;
          if (input.checked) nextDisabled.delete(name);
          else nextDisabled.add(name);
          saveCodexDisabledFiles(nextDisabled);
          refreshCodexLog();
        });
      });
    }
    setCodexModeUI();
  } catch (e) { /* leave as-is */ }
}

async function refreshCodexLog() {
  const pause = document.getElementById('codex-pause-toggle');
  if (pause && pause.checked) return;
  const streamFiles = selectedCodexStreamFiles();
  const fname = codexCurrentFile;
  if (codexStreamMode && streamFiles.length === 0) {
    const pre = document.getElementById('codex-log');
    const meta = document.getElementById('codex-meta');
    if (pre) pre.textContent = '(no active codex logs selected)';
    if (meta) meta.textContent = 'stream · 0 logs';
    return;
  }
  if (!codexStreamMode && !fname) return;
  try {
    const url = codexStreamMode
      ? '/api/codex-log-stream?files=' + encodeURIComponent(streamFiles.join(','))
      : '/api/codex-log?file=' + encodeURIComponent(fname);
    const data = await fetchJSON(url);
    const pre = document.getElementById('codex-log');
    const meta = document.getElementById('codex-meta');
    if (!pre) return;
    if (data && data.error) {
      pre.textContent = '(error: ' + data.error + ')';
      if (meta) meta.textContent = (codexStreamMode ? 'stream' : fname) + ' — error';
      return;
    }
    // Auto-scroll behavior fix: previously checking the auto-scroll
    // toggle force-snapped scroll to bottom on every poll, which broke
    // history navigation (user scrolls up to read, next poll yanks
    // them to bottom). New behavior: only snap to bottom when the user
    // is ALREADY at the bottom (sticky-bottom). Auto-scroll toggle
    // OFF disables sticky-bottom entirely (preserve scroll position
    // even if at bottom). Toggle ON enables sticky-bottom only.
    const tail = document.getElementById('codex-tail-toggle');
    const wantTail = !tail || tail.checked;
    const wasAtBottom =
      pre.scrollTop + pre.clientHeight >= pre.scrollHeight - 8;
    const prevTop = pre.scrollTop;
    pre.innerHTML = data.html || '';
    if (wantTail && wasAtBottom) {
      pre.scrollTop = pre.scrollHeight;
    } else {
      // innerHTML rewrite resets scrollTop to 0; restore user's position.
      pre.scrollTop = prevTop;
    }
    if (meta) {
      if (data.mode === 'stream') {
        const trunc = data.truncated ? ' · capped' : '';
        meta.textContent = `stream · ${data.files.length} logs · ${data.entries} entries${trunc}`;
      } else {
        const kb = (data.size / 1024).toFixed(0);
        const trunc = data.truncated ? ' (tail of last 4MB)' : '';
        meta.textContent = `${data.name} · ${kb}KB${trunc}`;
      }
    }
  } catch (e) { /* leave previous */ }
}

const codexFileSelect = document.getElementById('codex-file-select');
if (codexFileSelect) {
  codexFileSelect.addEventListener('change', (e) => {
    codexCurrentFile = e.target.value;
    if (codexCurrentFile) localStorage.setItem(CODEX_FILE_KEY, codexCurrentFile);
    refreshCodexLog();
  });
}
const codexStreamToggle = document.getElementById('codex-stream-toggle');
if (codexStreamToggle) {
  codexStreamToggle.addEventListener('change', (e) => {
    codexStreamMode = e.target.checked;
    localStorage.setItem(CODEX_MODE_KEY, codexStreamMode ? 'stream' : 'single');
    setCodexModeUI();
    refreshCodexLog();
  });
}

const codexShowAllToggle = document.getElementById('codex-show-all-toggle');
if (codexShowAllToggle) {
  codexShowAllToggle.addEventListener('change', (e) => {
    codexShowAll = e.target.checked;
    localStorage.setItem('oss-training-dashboard-codex-show-all', String(codexShowAll));
    refreshCodexFileList().then(refreshCodexLog);
  });
}

// Initial render + polling.
setCodexModeUI();
refreshCodexFileList().then(refreshCodexLog);
setInterval(() => { refreshCodexFileList(); }, 10_000);
setInterval(refreshCodexLog, CODEX_POLL_MS);
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------


def _matches_run_dir(path: Path) -> bool:
    return path.is_dir() and any(fnmatch.fnmatch(path.name, pat) for pat in RUN_DIR_PATTERNS)


def _safe_mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _run_last_modified(run_dir: Path) -> float:
    mtimes = [_safe_mtime(run_dir)]
    for name in ("metrics.json", "score_log.json"):
        mtimes.append(_safe_mtime(run_dir / name))
    ckpts = list(run_dir.glob("step-*.pt"))
    if ckpts:
        mtimes.append(max(_safe_mtime(p) for p in ckpts))
    viz_dir = run_dir / "viz"
    if viz_dir.is_dir():
        pngs = list(viz_dir.glob("step-*.png"))
        mtimes.append(_safe_mtime(viz_dir))
        if pngs:
            mtimes.append(max(_safe_mtime(p) for p in pngs))
    return max(mtimes)


def discover_runs(parent: Path) -> list[dict]:
    runs: list[dict] = []
    if not parent.is_dir():
        return runs
    for child in parent.iterdir():
        if not _matches_run_dir(child):
            continue
        viz_dir = child / "viz"
        runs.append({
            "name": child.name,
            "path": str(child.resolve()),
            "last_modified": _run_last_modified(child),
            "has_train_log": (child / "metrics.json").is_file(),
            "has_viz": viz_dir.is_dir() and any(viz_dir.glob("step-*.png")),
            "has_score_log": (child / "score_log.json").is_file(),
        })
    runs.sort(key=lambda r: (float(r["last_modified"]), r["name"]), reverse=True)
    return runs


class DashboardHandler(BaseHTTPRequestHandler):
    output_dir: Path = None  # type: ignore[assignment]
    log_file: Path = None    # type: ignore[assignment]

    def _send_json(self, payload, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_text(self, text: str, status: int = 200) -> None:
        body = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html: str) -> None:
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args, **kwargs) -> None:  # noqa: D401
        # Quiet the default access logger.
        return

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        try:
            if path in ("/", "/index.html"):
                self._send_html(HTML)
            elif path == "/api/runs":
                runs = discover_runs(self.output_dir.parent)
                default_run = runs[0]["name"] if runs else self.output_dir.name
                self._send_json({"runs": runs, "default_run": default_run})
            elif path == "/api/info":
                output_dir = self._resolve_request_output_dir(parsed.query)
                self._send_json(self._build_info(output_dir))
            elif path == "/api/metrics":
                output_dir = self._resolve_request_output_dir(parsed.query)
                data = self._read_json(output_dir, "metrics.json", default=[])
                if isinstance(data, dict) and isinstance(data.get("train"), list):
                    train = data["train"]
                    cap = 2000
                    if len(train) > cap:
                        step = max(1, len(train) // cap)
                        data = {**data, "train": train[::step][-cap:] + train[-1:]}
                self._send_json(data)
            elif path == "/api/score":
                output_dir = self._resolve_request_output_dir(parsed.query)
                data = self._read_json(output_dir, "score_log.json", default=None)
                if data is None:
                    metrics = self._read_json(output_dir, "metrics.json", default={})
                    data = metrics.get("score", []) if isinstance(metrics, dict) else []
                self._send_json(data)
            elif path == "/api/log":
                output_dir = self._resolve_request_output_dir(parsed.query)
                self._send_text(self._read_log_tail(output_dir, n_lines=200))
            elif path == "/api/codex-logs":
                self._send_json(self._list_codex_logs())
            elif path == "/api/codex-log":
                params = parse_qs(parsed.query, keep_blank_values=False)
                fname_values = params.get("file", [])
                fname = fname_values[-1] if fname_values else ""
                payload = self._read_codex_log(fname)
                self._send_json(payload)
            elif path == "/api/codex-log-stream":
                params = parse_qs(parsed.query, keep_blank_values=False)
                files_values = params.get("files", [])
                limit_values = params.get("limit", [])
                files_csv = files_values[-1] if files_values else ""
                limit = self._parse_int(limit_values[-1], CODEX_STREAM_ENTRY_LIMIT) if limit_values else CODEX_STREAM_ENTRY_LIMIT
                payload = self._read_codex_log_stream(files_csv, limit=limit)
                self._send_json(payload)
            elif path == "/api/viz":
                output_dir = self._resolve_request_output_dir(parsed.query)
                # Lists step-XXXXX.png files in <output_dir>/viz/, sorted ascending.
                viz_dir = output_dir / "viz"
                files: list[str] = []
                if viz_dir.is_dir():
                    files = sorted(p.name for p in viz_dir.glob("step-*.png"))
                self._send_json({"files": files})
            elif path.startswith("/viz/"):
                output_dir = self._resolve_request_output_dir(parsed.query)
                # Serve a single PNG from <output_dir>/viz/.
                fname = path[len("/viz/"):]
                if "/" in fname or ".." in fname or not fname.endswith(".png"):
                    self._send_text("bad path", status=400)
                else:
                    viz_path = output_dir / "viz" / fname
                    if not viz_path.is_file():
                        self._send_text("not found", status=404)
                    else:
                        body = viz_path.read_bytes()
                        self.send_response(200)
                        self.send_header("Content-Type", "image/png")
                        self.send_header("Content-Length", str(len(body)))
                        self.send_header("Cache-Control", "no-store")
                        self.end_headers()
                        self.wfile.write(body)
            else:
                self._send_text("not found", status=404)
        except PermissionError as e:
            self._send_json({"error": str(e)}, status=403)
        except FileNotFoundError as e:
            self._send_json({"error": str(e)}, status=404)
        except Exception as e:  # pragma: no cover — last-ditch error path
            self._send_text(f"server error: {e}", status=500)

    # ---- helpers ----

    def _resolve_request_output_dir(self, query: str) -> Path:
        params = parse_qs(query, keep_blank_values=False)
        run_values = params.get("run", [])
        run_name = run_values[-1] if run_values else None
        if run_name is None:
            runs = discover_runs(self.output_dir.parent)
            if runs:
                return (self.output_dir.parent / runs[0]["name"]).resolve()
            return self.output_dir

        if "/" in run_name or "\\" in run_name or run_name in ("", ".", ".."):
            raise PermissionError("denied run selector")
        candidate = (self.output_dir.parent / run_name).resolve()
        parent = self.output_dir.parent.resolve()
        try:
            candidate.relative_to(parent)
        except ValueError as e:
            raise PermissionError("denied run selector") from e
        if not _matches_run_dir(candidate):
            raise PermissionError("denied run selector")
        if not candidate.is_dir():
            raise FileNotFoundError(f"run not found: {run_name}")
        return candidate

    def _log_file_for_output_dir(self, output_dir: Path) -> Path:
        if output_dir == self.output_dir:
            return self.log_file
        candidates = [
            output_dir / "train.log",
            output_dir / "training.log",
            self.log_file.parent / f"{output_dir.name}.log",
            self.log_file.parent / f"{output_dir.name}.txt",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return output_dir / "train.log"

    def _read_json(self, output_dir: Path, name: str, default):
        path = output_dir / name
        if not path.exists():
            return default
        try:
            text = path.read_text()
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                # v6 writes metrics.json as JSON Lines for append-only
                # crash-safety. Expose the same shape the frontend already
                # accepts for legacy bare train-row arrays.
                rows = [
                    json.loads(line)
                    for line in text.splitlines()
                    if line.strip()
                ]
                return rows if rows else default
        except json.JSONDecodeError:
            # Mid-write race — just retry next poll.
            return default

    def _read_log_tail(self, output_dir: Path, n_lines: int) -> str:
        log_file = self._log_file_for_output_dir(output_dir)
        if not log_file.exists():
            return f"(log file not found: {log_file})"
        try:
            with log_file.open(encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
        except Exception as e:
            return f"(log read error: {e})"
        return "".join(lines[-n_lines:])

    # ---- codex live-log helpers ----

    @staticmethod
    def _parse_int(value: str, default: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def _list_codex_logs(self) -> dict:
        """Return summary of /tmp/codex-*.log files (name, size, mtime, alive)."""
        if not CODEX_LOG_DIR.is_dir():
            return {"files": []}
        try:
            paths = sorted(CODEX_LOG_DIR.glob(CODEX_LOG_GLOB),
                           key=lambda p: p.stat().st_mtime, reverse=True)
        except Exception:
            return {"files": []}
        # Active codex PIDs (best-effort, macOS / linux compatible).
        active_cmdlines = self._codex_active_cmdlines()
        files = []
        for p in paths:
            try:
                st = p.stat()
            except Exception:
                continue
            stem = p.stem  # e.g. "codex-v6model-stage2"
            alive = any(stem in cmd for cmd in active_cmdlines)
            active = (time.time() - st.st_mtime) <= CODEX_ACTIVE_SECONDS
            files.append({
                "name": p.name,
                "size": int(st.st_size),
                "mtime": st.st_mtime,
                "alive": alive,
                "active": active,
            })
        return {"files": files}

    def _codex_active_cmdlines(self) -> list[str]:
        """Best-effort list of running codex-exec command lines (used to flag
        which logs map to a still-running process). Returns [] on failure."""
        try:
            import subprocess
            res = subprocess.run(
                ["pgrep", "-fl", "codex exec"],
                capture_output=True, text=True, timeout=2,
            )
            if res.returncode != 0:
                return []
            return [line.strip() for line in res.stdout.splitlines() if line.strip()]
        except Exception:
            return []

    def _validate_codex_log_name(self, fname: str) -> str | None:
        if not fname or "/" in fname or "\\" in fname or ".." in fname:
            return "bad filename"
        if not fname.startswith("codex-") or not fname.endswith(".log"):
            return "not a codex log"
        return None

    def _read_codex_log_text(self, fname: str) -> dict:
        """Return raw text of a /tmp/codex-*.log file capped to last 4 MB."""
        err = self._validate_codex_log_name(fname)
        if err:
            return {"error": err}
        path = CODEX_LOG_DIR / fname
        if not path.is_file():
            return {"error": "not found"}
        try:
            st = path.stat()
            size = int(st.st_size)
            with path.open("rb") as f:
                if size > CODEX_LOG_CAP_BYTES:
                    f.seek(size - CODEX_LOG_CAP_BYTES)
                    raw = f.read()
                    text = raw.decode("utf-8", errors="replace")
                    # The state machine only renders correctly if it starts
                    # at a known mode boundary. Walk forward to the next
                    # 'codex' / 'exec' line; everything before it is
                    # pre-truncation context that would render mid-stream.
                    lines = text.splitlines(keepends=True)
                    start = 0
                    for idx, ln in enumerate(lines):
                        s = ln.rstrip("\n")
                        if s == "codex" or s == "exec":
                            start = idx
                            break
                    text = "".join(lines[start:])
                    truncated = True
                else:
                    text = f.read().decode("utf-8", errors="replace")
                    truncated = False
        except Exception as e:
            return {"error": f"read error: {e}"}
        return {
            "name": fname,
            "size": size,
            "mtime": st.st_mtime,
            "truncated": truncated,
            "text": text,
        }

    def _read_codex_log(self, fname: str) -> dict:
        """Return rendered HTML of one /tmp/codex-*.log file."""
        payload = self._read_codex_log_text(fname)
        if "error" in payload:
            return payload
        text = str(payload["text"])
        if _render_codex_html is not None:
            html = _render_codex_html(text, keep_mcp=False)
        else:
            html = _html.escape(text)
        return {
            "name": payload["name"],
            "size": payload["size"],
            "mtime": payload["mtime"],
            "truncated": payload["truncated"],
            "html": html,
        }

    def _read_codex_log_stream(self, files_csv: str, *, limit: int) -> dict:
        """Return recent codex/exec entries merged by timestamp across logs."""
        limit = max(1, min(limit, 500))
        requested = [f for f in files_csv.split(",") if f]
        if not requested:
            requested = [
                f["name"] for f in self._list_codex_logs()["files"]
                if f.get("active")
            ]

        entries: list[dict] = []
        used_files: list[str] = []
        truncated = False
        now = time.time()
        for fname in requested:
            err = self._validate_codex_log_name(fname)
            if err:
                return {"error": err}
            path = CODEX_LOG_DIR / fname
            if not path.is_file():
                continue
            try:
                st = path.stat()
            except OSError:
                continue
            if now - st.st_mtime > CODEX_ACTIVE_SECONDS:
                continue
            payload = self._read_codex_log_text(fname)
            if "error" in payload:
                continue
            used_files.append(fname)
            truncated = truncated or bool(payload["truncated"])
            entries.extend(
                self._codex_log_entries(
                    fname,
                    str(payload["text"]),
                    float(payload["mtime"]),
                )
            )

        entries = sorted(entries, key=lambda e: (float(e["timestamp"]), str(e["file"]), int(e["index"])))
        entries = entries[-limit:]
        html_parts = []
        for entry in entries:
            source = _html.escape(str(entry["file"]), quote=False)
            html_parts.append(f'<div class="codex-entry-source">{source}</div>')
            body = str(entry["text"])
            if _render_codex_html is not None:
                html_parts.append(_render_codex_html(body, keep_mcp=False))
            else:
                html_parts.append(_html.escape(body))
        return {
            "mode": "stream",
            "files": used_files,
            "entries": len(entries),
            "truncated": truncated,
            "html": "\n".join(part for part in html_parts if part),
        }

    def _codex_log_entries(self, fname: str, text: str, mtime: float) -> list[dict]:
        chunks: list[list[str]] = []
        current: list[str] = []
        for line in text.splitlines():
            if line in ("user", "codex", "exec", "apply patch") and current:
                chunks.append(current)
                current = [line]
            else:
                current.append(line)
        if current:
            chunks.append(current)

        total = max(1, len(chunks))
        entries = []
        for idx, lines in enumerate(chunks):
            timestamp = self._entry_timestamp(lines)
            if timestamp is None:
                timestamp = mtime - ((total - idx) / 1000.0)
            entries.append({
                "file": fname,
                "index": idx,
                "timestamp": timestamp,
                "text": "\n".join(lines),
            })
        return entries

    @staticmethod
    def _entry_timestamp(lines: list[str]) -> float | None:
        for line in lines:
            m = CODEX_TIMESTAMP_RE.match(line)
            if not m:
                continue
            try:
                from datetime import datetime, timezone

                dt = datetime.fromisoformat(m.group(1).replace("Z", "+00:00"))
                return dt.astimezone(timezone.utc).timestamp()
            except ValueError:
                return None
        return None

    def _build_info(self, output_dir: Path) -> dict:
        log_file = self._log_file_for_output_dir(output_dir)
        info: dict = {
            "output_dir": str(output_dir),
            "log_file": str(log_file),
            "run": output_dir.name,
        }
        # Try to surface max_steps + max_time_seconds + score_every from the latest checkpoint args.
        ckpts = sorted(output_dir.glob("step-*.pt")) if output_dir.exists() else []
        if ckpts:
            try:
                import torch  # type: ignore[import-not-found]
                ck = torch.load(ckpts[-1], map_location="cpu", weights_only=False)
                args = ck.get("args", {})
                info["max_steps"] = int(args.get("max_steps", 0)) or None
                info["max_time_seconds"] = args.get("max_time_seconds")
                info["eval_every"] = int(args.get("score_every", 5000))
                info["tier"] = args.get("tier")
                info["model_kind"] = args.get("model_kind")
            except Exception:
                pass
        # Log timestamps for liveness.
        if log_file.exists():
            stat = log_file.stat()
            info["last_log_mtime_unix"] = stat.st_mtime

        # Try to estimate elapsed from log first/last timestamps.
        if log_file.exists():
            try:
                first_line = None
                last_line = None
                with log_file.open(encoding="utf-8", errors="replace") as f:
                    for line in f:
                        if first_line is None and line.startswith("20"):
                            first_line = line
                        if line.startswith("20"):
                            last_line = line
                if first_line and last_line:
                    import datetime as _dt
                    def parse_ts(s: str):
                        ts = s.split(",")[0]
                        return _dt.datetime.strptime(ts, "%Y-%m-%d %H:%M:%S").timestamp()
                    start = parse_ts(first_line)
                    last = parse_ts(last_line)
                    info["start_time_unix"] = start
                    info["last_log_time_unix"] = last
                    info["elapsed_seconds"] = max(0.0, last - start)
            except Exception:
                pass

        return info


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True,
                   help="Training output dir (where metrics.json, score_log.json, step-*.pt live).")
    p.add_argument("--log-file", type=Path, required=True,
                   help="Tee'd training log file path.")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--host", type=str, default="0.0.0.0")
    args = p.parse_args()

    DashboardHandler.output_dir = args.output_dir.resolve()
    DashboardHandler.log_file = args.log_file.resolve()

    print(f"Dashboard: http://{args.host}:{args.port}/")
    print(f"  output_dir = {DashboardHandler.output_dir}")
    print(f"  log_file   = {DashboardHandler.log_file}")

    class ThreadedServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
        allow_reuse_address = True
        daemon_threads = True

    with ThreadedServer((args.host, args.port), DashboardHandler) as srv:
        srv.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
