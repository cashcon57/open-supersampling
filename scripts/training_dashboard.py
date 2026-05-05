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
import json
import socketserver
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse

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
  .grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; }
  .panel { background: var(--panel); border: 1px solid var(--border);
           border-radius: 8px; padding: 16px; }
  .panel h2 { font-size: 14px; margin: 0 0 12px 0; color: var(--muted);
              text-transform: uppercase; letter-spacing: 0.5px; }
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

<h1><span class="status-dot" id="status-dot"></span>OSS Training Dashboard</h1>
<div class="sub" id="header-sub">loading…</div>

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
    <h2>PSNR <span style="font-size:13px;color:#3fb950">↑ higher is better</span></h2>
    <canvas id="chart-psnr"></canvas>
    <div style="font-size:11px;color:#8b949e;margin-top:4px"><b>Drag to pan · scroll to zoom · double-click to reset.</b> Solid line: live training-time PSNR proxy (≈ −10·log10(t_l1²)). Held-out eval lines populate after `sr_temporal_held_out.py` runs (closeout). Dashed: published-benchmark estimates of competing upscalers at 1080p→4K Quality (±1 dB envelope). Solid + thicker = our measured numbers (OSS v3, v4).</div>
  </div>

  <div class="panel">
    <h2>LPIPS-VGG <span style="font-size:13px;color:#3fb950">↓ lower is better</span></h2>
    <canvas id="chart-lpips"></canvas>
    <div style="font-size:11px;color:#8b949e;margin-top:4px">Solid line: live training-time LPIPS (Phase 2+ only, when LPIPS loss is enabled). Held-out eval lines populate after closeout. Dashed: published-benchmark estimates (Bicubic ≈ 0.51, DLSS 2/FSR 2 ≈ 0.22, DLSS 4 ≈ 0.17).</div>
  </div>

  <div class="panel">
    <h2>Throughput <span style="font-size:13px;color:#3fb950">↑ higher is better (steps/min)</span></h2>
    <canvas id="chart-throughput"></canvas>
    <div style="font-size:11px;color:#8b949e;margin-top:4px">Computed from train-row timestamps. Sustained drop indicates DataLoader starvation or compute-bound phase (Phase 2 LPIPS-VGG cuts throughput ~5×).</div>
  </div>

  <div class="panel">
    <h2>Loss decomposition <span style="font-size:13px;color:#3fb950">↓ lower is better</span></h2>
    <canvas id="chart-loss"></canvas>
    <div style="font-size:11px;color:#8b949e;margin-top:4px">Phase 1: appearance loss (L1+SSIM) only. Phase 2 (step 10K+): adds LPIPS + temporal-consistency. Phase 3 (step 60K+): same loss, LR×0.01 polish.</div>
  </div>

  <div class="panel">
    <h2>SSIM <span style="font-size:13px;color:#3fb950">↑ higher is better</span></h2>
    <canvas id="chart-ssim"></canvas>
  </div>

  <div class="panel">
    <h2>Output stats (mean, std)</h2>
    <canvas id="chart-out"></canvas>
  </div>

  <div class="panel">
    <h2>Gradient norms</h2>
    <canvas id="chart-grad"></canvas>
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

</div>

<script>
const POLL_MS = 10_000;
const STALE_MS = 60_000;

let charts = {};
let lastUpdate = null;

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
        legend: { labels: { color: '#e6edf3' } },
        // chartjs-plugin-zoom: scroll-wheel zoom (cursor-anchored), drag to
        // pan the x-axis, double-click to reset. Vertical zoom enabled too
        // so users can magnify into a tight loss range.
        zoom: {
          pan: { enabled: true, mode: 'xy', modifierKey: null },
          zoom: {
            wheel: { enabled: true, speed: 0.1 },
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
             ticks: { color: '#8b949e' }, grid: { color: '#30363d' } },
        y: { title: { display: true, text: opts.yLabel || label, color: '#8b949e' },
             ticks: { color: '#8b949e' }, grid: { color: '#30363d' },
             ...(opts.yMin !== undefined ? { min: opts.yMin } : {}),
             ...(opts.yMax !== undefined ? { max: opts.yMax } : {}) },
      },
    },
  });
}

function setChart(chart, datasets) {
  chart.data.datasets = datasets;
  chart.update();
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
    for (const [k, label] of items) {
      const e = UPSCALER_ESTIMATES[k];
      const v = e[metric];
      if (v == null) continue;
      const yp = y.getPixelForValue(v);
      if (yp < y.top || yp > y.bottom) continue;
      // OSS lines are solid + thicker so 'ours' reads distinctly from competitors.
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
      ctx.fillStyle = e.color;
      const suffix = e.kind === 'ours' ? '' : ' (est)';
      ctx.fillText(label + suffix, x.right - 110, yp - 2);
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
  charts.psnr = lineChart('chart-psnr', 'PSNR (dB)', null, {
    yLabel: 'PSNR (dB) ↑ better', yMin: 22, yMax: 36,
    extraPlugins: [phaseMarkerPlugin, upscalerRefPlugin('psnr')],
  });
  charts.lpips = lineChart('chart-lpips', 'LPIPS', null, {
    yLabel: 'LPIPS ↓ better', yMin: 0, yMax: 0.6,
    extraPlugins: [phaseMarkerPlugin, upscalerRefPlugin('lpips')],
  });
  charts.throughput = lineChart('chart-throughput', 'steps/min', null, {
    yLabel: 'steps/min ↑ better', yMin: 0,
    extraPlugins: [phaseMarkerPlugin],
  });
  charts.loss = lineChart('chart-loss', 'loss', null, {
    yLabel: 'loss ↓ better',
    extraPlugins: [phaseMarkerPlugin],
  });
  charts.ssim = lineChart('chart-ssim', 'SSIM', null, {
    yLabel: 'SSIM ↑ better', yMin: 0, yMax: 1,
    extraPlugins: [phaseMarkerPlugin],
  });
  charts.out = lineChart('chart-out', 'output');
  charts.grad = lineChart('chart-grad', 'grad norm');
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
  img.src = '/viz/' + fname + '?_t=' + Date.now();
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
      fetchJSON('/api/info'),
      fetchJSON('/api/metrics'),
      fetchJSON('/api/score'),
      fetchText('/api/log'),
      fetchJSON('/api/viz').catch(() => ({ files: [] })),
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
refresh();
setInterval(refresh, POLL_MS);
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------


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
        path = urlparse(self.path).path
        try:
            if path in ("/", "/index.html"):
                self._send_html(HTML)
            elif path == "/api/info":
                self._send_json(self._build_info())
            elif path == "/api/metrics":
                data = self._read_json("metrics.json", default=[])
                if isinstance(data, dict) and isinstance(data.get("train"), list):
                    train = data["train"]
                    cap = 2000
                    if len(train) > cap:
                        step = max(1, len(train) // cap)
                        data = {**data, "train": train[::step][-cap:] + train[-1:]}
                self._send_json(data)
            elif path == "/api/score":
                data = self._read_json("score_log.json", default=None)
                if data is None:
                    metrics = self._read_json("metrics.json", default={})
                    data = metrics.get("score", []) if isinstance(metrics, dict) else []
                self._send_json(data)
            elif path == "/api/log":
                self._send_text(self._read_log_tail(n_lines=200))
            elif path == "/api/viz":
                # Lists step-XXXXX.png files in <output_dir>/viz/, sorted ascending.
                viz_dir = self.output_dir / "viz"
                files: list[str] = []
                if viz_dir.is_dir():
                    files = sorted(p.name for p in viz_dir.glob("step-*.png"))
                self._send_json({"files": files})
            elif path.startswith("/viz/"):
                # Serve a single PNG from <output_dir>/viz/.
                fname = path[len("/viz/"):]
                if "/" in fname or ".." in fname or not fname.endswith(".png"):
                    self._send_text("bad path", status=400)
                else:
                    viz_path = self.output_dir / "viz" / fname
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
        except Exception as e:  # pragma: no cover — last-ditch error path
            self._send_text(f"server error: {e}", status=500)

    # ---- helpers ----

    def _read_json(self, name: str, default):
        path = self.output_dir / name
        if not path.exists():
            return default
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            # Mid-write race — just retry next poll.
            return default

    def _read_log_tail(self, n_lines: int) -> str:
        if not self.log_file.exists():
            return f"(log file not found: {self.log_file})"
        try:
            with self.log_file.open(encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
        except Exception as e:
            return f"(log read error: {e})"
        return "".join(lines[-n_lines:])

    def _build_info(self) -> dict:
        info: dict = {
            "output_dir": str(self.output_dir),
            "log_file": str(self.log_file),
        }
        # Try to surface max_steps + max_time_seconds + score_every from the latest checkpoint args.
        ckpts = sorted(self.output_dir.glob("step-*.pt")) if self.output_dir.exists() else []
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
        if self.log_file.exists():
            stat = self.log_file.stat()
            info["last_log_mtime_unix"] = stat.st_mtime

        # Try to estimate elapsed from log first/last timestamps.
        if self.log_file.exists():
            try:
                first_line = None
                last_line = None
                with self.log_file.open(encoding="utf-8", errors="replace") as f:
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
