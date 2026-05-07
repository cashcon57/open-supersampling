#!/usr/bin/env python3
"""Build the static public training dashboard.

Input lives under dashboard-public/runs/<run-name>/ and is restored from the
gh-pages branch before the GitHub Action fetches fresh training-host files.
The output is a static index.html plus data.json. No third-party Python deps.
"""

from __future__ import annotations

import datetime as _dt
import html
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_DIR = ROOT / "dashboard-public"
RUNS_DIR = PUBLIC_DIR / "runs"
README = ROOT / "README.md"
DATA_JSON = PUBLIC_DIR / "data.json"
INDEX_HTML = PUBLIC_DIR / "index.html"

RUN_CONFIG = {
    "srcnn-v6.1-pico-001": {"label": "v6.1 Pico (active)", "active": True},
    "srcnn-v5-pixel-temporal-validated": {
        "label": "v5 Pixel Temporal (validated)",
        "active": False,
    },
    "srcnn-v6-pico-001": {"label": "v6 Pico (historical)", "active": False},
}

RUN_ORDER = list(RUN_CONFIG)
DENY_RE = re.compile(
    r"(aborted|smoke|sanity|test|leak|preflight|paramprobe|init-fix|diag)",
    re.IGNORECASE,
)
VIZ_RE = re.compile(r"step-(\d+)\.png$")


def utc_now_iso() -> str:
    return _dt.datetime.now(_dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def read_rows(path: Path) -> list[dict[str, Any]]:
    """Read a JSON array, a dict with train rows, or JSONL rows."""

    if not path.is_file():
        return []
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        return []

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = None

    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        rows = payload.get("train") or payload.get("rows") or payload.get("metrics")
        if isinstance(rows, list):
            return [row for row in rows if isinstance(row, dict)]
        return [payload]

    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            # Ignore torn final JSONL writes from an interrupted rsync.
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def primitive_value(value: Any) -> Any:
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float, str)):
        return value
    return None


def slim_row(row: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in row.items():
        primitive = primitive_value(value)
        if primitive is not None or value is None:
            out[str(key)] = primitive
    return out


def step_value(row: dict[str, Any]) -> int:
    raw = row.get("step", 0)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return 0


def viz_step(name: str) -> int:
    match = VIZ_RE.match(name)
    return int(match.group(1)) if match else -1


def list_viz_pngs(run_dir: Path) -> list[str]:
    viz_dir = run_dir / "viz"
    if not viz_dir.is_dir():
        return []
    return sorted((path.name for path in viz_dir.glob("step-*.png")), key=viz_step)


def load_previous_runs() -> dict[str, dict[str, Any]]:
    if not DATA_JSON.is_file():
        return {}
    try:
        data = json.loads(DATA_JSON.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    runs = data.get("runs")
    if not isinstance(runs, list):
        return {}
    return {str(run.get("name")): run for run in runs if isinstance(run, dict) and run.get("name")}


def build_run(name: str, previous: dict[str, Any] | None) -> dict[str, Any] | None:
    if DENY_RE.search(name) or name not in RUN_CONFIG:
        return None

    config = RUN_CONFIG[name]
    run_dir = RUNS_DIR / name
    metrics = read_rows(run_dir / "metrics.json")
    scores = read_rows(run_dir / "score_log.json")
    viz_pngs = list_viz_pngs(run_dir)

    previous_has_data = bool(
        previous
        and (
            previous.get("latest_step")
            or previous.get("loss_curve")
            or previous.get("score_log")
            or previous.get("viz_pngs")
        )
    )
    if not metrics and not scores and not viz_pngs and previous_has_data:
        cached = dict(previous)
        cached["cached"] = True
        cached["label"] = config["label"]
        cached["active"] = config["active"]
        return cached

    if not metrics and not scores and not viz_pngs:
        return {
            "name": name,
            "label": config["label"],
            "active": config["active"],
            "latest_step": 0,
            "latest_metrics": {},
            "loss_curve": [],
            "score_log": [],
            "viz_pngs": [],
        }

    metrics = sorted(metrics, key=step_value)
    latest = slim_row(metrics[-1]) if metrics else {}
    latest_step = step_value(metrics[-1]) if metrics else 0

    return {
        "name": name,
        "label": config["label"],
        "active": config["active"],
        "latest_step": latest_step,
        "latest_metrics": latest,
        "loss_curve": [slim_row(row) for row in metrics[-1000:]],
        "score_log": [slim_row(row) for row in scores],
        "viz_pngs": viz_pngs,
    }


def extract_pitch() -> str:
    if not README.is_file():
        return (
            "OSS uses one persistent Gaussian canvas for game super-resolution and "
            "frame extrapolation. The same canvas can render the current frame or a "
            "fractional future frame without a second frame-generation network."
        )

    text = README.read_text(encoding="utf-8", errors="replace")
    marker = "## Why this architecture"
    start = text.find(marker)
    if start < 0:
        return ""
    rest = text[start + len(marker) :]
    paragraphs = [part.strip() for part in rest.split("\n\n") if part.strip()]
    if not paragraphs:
        return ""
    return paragraphs[0].replace("\n", " ")


def build_data() -> dict[str, Any]:
    PUBLIC_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    previous = load_previous_runs()
    runs = []
    for name in RUN_ORDER:
        run = build_run(name, previous.get(name))
        if run is not None:
            runs.append(run)

    return {
        "generated_at": utc_now_iso(),
        "runs": runs,
    }


def write_data(data: dict[str, Any]) -> None:
    DATA_JSON.write_text(
        json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_index(pitch: str) -> None:
    pitch_html = html.escape(pitch)
    index = HTML_TEMPLATE.replace("__PITCH_HTML__", pitch_html)
    INDEX_HTML.write_text(index, encoding="utf-8")


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en" class="dark">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>OSS Training Dashboard</title>
  <link rel="icon" href="data:,">
  <script src="https://cdn.tailwindcss.com"></script>
  <script>
    tailwind.config = { darkMode: "class" };
  </script>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0"></script>
  <style>
    .viz-strip { scrollbar-width: thin; }
    .chart-wrap { height: 18rem; }
    .light-mode [class*="bg-zinc-950"] { background-color: #f8fafc !important; }
    .light-mode [class*="bg-zinc-900"] { background-color: rgba(255, 255, 255, 0.92) !important; }
    .light-mode [class*="border-zinc-800"],
    .light-mode [class*="border-zinc-700"] { border-color: #d4d4d8 !important; }
    .light-mode [class*="text-zinc-50"],
    .light-mode [class*="text-zinc-100"] { color: #18181b !important; }
    .light-mode [class*="text-zinc-200"],
    .light-mode [class*="text-zinc-300"],
    .light-mode [class*="text-zinc-400"] { color: #3f3f46 !important; }
    .light-mode [class*="text-zinc-500"] { color: #71717a !important; }
    .light-mode select { background-color: #ffffff !important; color: #18181b !important; }
    @media (max-width: 640px) { .chart-wrap { height: 16rem; } }
  </style>
</head>
<body class="min-h-screen bg-zinc-950 text-zinc-100 antialiased transition-colors dark:bg-zinc-950 dark:text-zinc-100">
  <main class="mx-auto flex max-w-7xl flex-col gap-6 px-4 py-5 sm:px-6 lg:px-8">
    <header class="border-b border-zinc-800 pb-5 dark:border-zinc-800">
      <div class="flex flex-col gap-4 md:flex-row md:items-start md:justify-between">
        <div class="max-w-4xl">
          <p class="text-xs font-semibold uppercase tracking-wide text-sky-400">OpenSuperSampling</p>
          <h1 class="mt-2 text-3xl font-semibold tracking-normal text-zinc-50 sm:text-4xl">Public training dashboard</h1>
          <p class="mt-3 max-w-3xl text-sm leading-6 text-zinc-300">__PITCH_HTML__</p>
          <div class="mt-4 flex flex-wrap items-center gap-3 text-sm text-zinc-400">
            <a class="font-medium text-sky-400 hover:text-sky-300" href="https://github.com/cashcon57/open-supersampling">GitHub repo</a>
            <span class="hidden text-zinc-700 sm:inline">/</span>
            <span id="updated-line">updated just now</span>
            <span id="data-state" class="rounded border border-zinc-700 px-2 py-0.5 text-xs text-zinc-300">loading</span>
          </div>
        </div>
        <button id="theme-toggle" class="inline-flex h-9 items-center justify-center rounded-md border border-zinc-700 px-3 text-sm text-zinc-200 hover:border-sky-500 hover:text-white" type="button">Light mode</button>
      </div>
    </header>

    <nav class="flex gap-2 overflow-x-auto border-b border-zinc-800 pb-2" aria-label="Dashboard sections">
      <button class="tab-btn rounded-md bg-sky-500 px-3 py-2 text-sm font-medium text-white" data-tab="active" type="button">Active runs</button>
      <button class="tab-btn rounded-md px-3 py-2 text-sm font-medium text-zinc-300 hover:bg-zinc-900" data-tab="result" type="button">Latest measured result</button>
      <button class="tab-btn rounded-md px-3 py-2 text-sm font-medium text-zinc-300 hover:bg-zinc-900" data-tab="architecture" type="button">Architecture</button>
    </nav>

    <section id="tab-active" class="tab-panel flex flex-col gap-5">
      <div class="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h2 class="text-xl font-semibold text-zinc-50">Active v6.1 training signal</h2>
          <p class="mt-1 text-sm text-zinc-400">Loss curves, held-out eval rows, and the latest in-flight visualization strips.</p>
        </div>
        <label class="flex items-center gap-2 text-sm text-zinc-400">
          <span>run</span>
          <select id="run-select" class="rounded-md border border-zinc-700 bg-zinc-950 px-3 py-2 font-mono text-xs text-zinc-100"></select>
        </label>
      </div>

      <div id="active-empty" class="hidden rounded-md border border-zinc-800 bg-zinc-900/60 p-5 text-sm text-zinc-300">
        No active run data has been published yet. Once the operator configures the Tailscale secrets, this page will populate from the 3080 Ti training host.
      </div>

      <div id="active-content" class="grid gap-5 lg:grid-cols-2">
        <article class="rounded-md border border-zinc-800 bg-zinc-900/60 p-4">
          <div class="flex items-start justify-between gap-4">
            <div>
              <h3 id="active-title" class="text-base font-semibold text-zinc-50">Run</h3>
              <p id="active-meta" class="mt-1 text-sm text-zinc-400">waiting for data</p>
            </div>
            <div class="grid grid-cols-2 gap-2 text-right text-xs">
              <div class="rounded border border-zinc-800 px-3 py-2">
                <div class="text-zinc-500">step</div>
                <div id="active-step" class="font-mono text-base text-zinc-100">0</div>
              </div>
              <div class="rounded border border-zinc-800 px-3 py-2">
                <div class="text-zinc-500">loss</div>
                <div id="active-loss" class="font-mono text-base text-zinc-100">--</div>
              </div>
            </div>
          </div>
          <div class="chart-wrap mt-4">
            <canvas id="loss-chart"></canvas>
          </div>
        </article>

        <article class="rounded-md border border-zinc-800 bg-zinc-900/60 p-4">
          <h3 class="text-base font-semibold text-zinc-50">Held-out eval</h3>
          <p id="score-meta" class="mt-1 text-sm text-zinc-400">PSNR and LPIPS populate when score_log.json lands.</p>
          <div class="chart-wrap mt-4">
            <canvas id="score-chart"></canvas>
          </div>
        </article>

        <article class="rounded-md border border-zinc-800 bg-zinc-900/60 p-4 lg:col-span-2">
          <div class="flex items-center justify-between gap-4">
            <h3 class="text-base font-semibold text-zinc-50">Visualization strip</h3>
            <span id="viz-count" class="text-sm text-zinc-400">0 images</span>
          </div>
          <div id="viz-strip" class="viz-strip mt-4 flex gap-3 overflow-x-auto pb-2"></div>
        </article>
      </div>
    </section>

    <section id="tab-result" class="tab-panel hidden flex-col gap-5">
      <div>
        <h2 class="text-xl font-semibold text-zinc-50">Latest measured result</h2>
        <p class="mt-1 text-sm text-zinc-400">v5-pixel-temporal final held-out eval on TartanAir oldtown.</p>
      </div>
      <div class="grid gap-4 sm:grid-cols-3">
        <div class="rounded-md border border-zinc-800 bg-zinc-900/60 p-5">
          <div class="text-sm uppercase text-zinc-500">PSNR</div>
          <div class="mt-2 font-mono text-3xl font-semibold text-emerald-400">25.703</div>
          <div class="mt-1 text-sm text-zinc-400">dB, higher is better</div>
        </div>
        <div class="rounded-md border border-zinc-800 bg-zinc-900/60 p-5">
          <div class="text-sm uppercase text-zinc-500">LPIPS</div>
          <div class="mt-2 font-mono text-3xl font-semibold text-emerald-400">0.1666</div>
          <div class="mt-1 text-sm text-zinc-400">lower is better</div>
        </div>
        <div class="rounded-md border border-zinc-800 bg-zinc-900/60 p-5">
          <div class="text-sm uppercase text-zinc-500">Temporal ratio</div>
          <div class="mt-2 font-mono text-3xl font-semibold text-emerald-400">0.337x</div>
          <div class="mt-1 text-sm text-zinc-400">versus v4 baseline</div>
        </div>
      </div>
      <article class="rounded-md border border-zinc-800 bg-zinc-900/60 p-4">
        <div class="flex items-center justify-between gap-4">
          <h3 class="text-base font-semibold text-zinc-50">v5 examples</h3>
          <span id="result-viz-count" class="text-sm text-zinc-400">0 images</span>
        </div>
        <div id="result-viz-strip" class="viz-strip mt-4 flex gap-3 overflow-x-auto pb-2"></div>
      </article>
    </section>

    <section id="tab-architecture" class="tab-panel hidden flex-col gap-5">
      <div class="rounded-md border border-zinc-800 bg-zinc-900/60 p-5">
        <h2 class="text-xl font-semibold text-zinc-50">Architecture memo</h2>
        <p class="mt-2 max-w-3xl text-sm leading-6 text-zinc-300">The canonical v6 design is the source of truth for the persistent Gaussian canvas, covariance-resampled rasterizer, cross-attention fusion path, and OSS-FX frame extrapolation plan.</p>
        <a class="mt-4 inline-flex rounded-md border border-sky-500 px-3 py-2 text-sm font-medium text-sky-300 hover:bg-sky-500 hover:text-white" href="https://github.com/cashcon57/open-supersampling/blob/main/docs/superpowers/experiments/2026-05-05-v6-architecture-canonical.md">Open canonical memo</a>
      </div>
    </section>
  </main>

<script>
"use strict";

let dashboardData = null;
let lossChart = null;
let scoreChart = null;

const colors = {
  total: "#38bdf8",
  charbonnier: "#f59e0b",
  lpips: "#f43f5e",
  psnr: "#34d399",
  bicubic: "#94a3b8",
};

function fmtNumber(value, digits = 4) {
  if (value === undefined || value === null || Number.isNaN(Number(value))) return "--";
  const num = Number(value);
  if (Math.abs(num) >= 1000) return num.toLocaleString();
  return num.toFixed(digits).replace(/\.?0+$/, "");
}

function stepFromName(name) {
  const match = /step-(\d+)\.png$/.exec(name || "");
  return match ? Number(match[1]) : 0;
}

function agoText(iso) {
  const ts = Date.parse(iso || "");
  if (!Number.isFinite(ts)) return "updated unknown";
  const seconds = Math.max(0, Math.floor((Date.now() - ts) / 1000));
  if (seconds < 60) return "updated just now";
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `updated ${minutes} min ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 48) return `updated ${hours} hr ago`;
  return `updated ${Math.floor(hours / 24)} days ago`;
}

function runUrl(run, file) {
  return `runs/${encodeURIComponent(run.name)}/viz/${encodeURIComponent(file)}`;
}

function setTheme(light) {
  document.documentElement.classList.toggle("dark", !light);
  document.body.classList.toggle("light-mode", light);
  document.body.classList.toggle("bg-white", light);
  document.body.classList.toggle("text-zinc-950", light);
  document.body.classList.toggle("bg-zinc-950", !light);
  document.body.classList.toggle("text-zinc-100", !light);
  localStorage.setItem("oss-dashboard-theme", light ? "light" : "dark");
  document.getElementById("theme-toggle").textContent = light ? "Dark mode" : "Light mode";
}

function setupTabs() {
  document.querySelectorAll(".tab-btn").forEach((button) => {
    button.addEventListener("click", () => {
      const tab = button.dataset.tab;
      document.querySelectorAll(".tab-btn").forEach((item) => {
        const active = item.dataset.tab === tab;
        item.classList.toggle("bg-sky-500", active);
        item.classList.toggle("text-white", active);
        item.classList.toggle("text-zinc-300", !active);
      });
      document.querySelectorAll(".tab-panel").forEach((panel) => {
        panel.classList.toggle("hidden", panel.id !== `tab-${tab}`);
        panel.classList.toggle("flex", panel.id === `tab-${tab}`);
      });
    });
  });
}

function makeLineChart(canvasId, yTitle) {
  const ctx = document.getElementById(canvasId);
  return new Chart(ctx, {
    type: "line",
    data: { datasets: [] },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      parsing: false,
      interaction: { mode: "nearest", intersect: false },
      scales: {
        x: {
          type: "linear",
          title: { display: true, text: "step", color: "#a1a1aa" },
          grid: { color: "rgba(113,113,122,0.22)" },
          ticks: { color: "#a1a1aa" },
        },
        y: {
          title: { display: true, text: yTitle, color: "#a1a1aa" },
          grid: { color: "rgba(113,113,122,0.22)" },
          ticks: { color: "#a1a1aa" },
        },
      },
      plugins: {
        legend: { labels: { color: "#d4d4d8", boxWidth: 12 } },
      },
    },
  });
}

function xy(rows, key, ...aliases) {
  return (rows || []).map((row) => {
    let value = row[key];
    for (const alias of aliases) {
      if (value !== undefined && value !== null) break;
      value = row[alias];
    }
    if (value === undefined || value === null) return null;
    return { x: Number(row.step || 0), y: Number(value) };
  }).filter(Boolean);
}

function renderVizStrip(hostId, countId, run, limit) {
  const host = document.getElementById(hostId);
  const count = document.getElementById(countId);
  host.replaceChildren();

  const files = (run && run.viz_pngs ? run.viz_pngs : []).slice(-limit).reverse();
  count.textContent = `${files.length} image${files.length === 1 ? "" : "s"}`;
  if (!files.length) {
    const empty = document.createElement("div");
    empty.className = "rounded border border-zinc-800 px-4 py-6 text-sm text-zinc-400";
    empty.textContent = "No visualization PNGs published yet.";
    host.appendChild(empty);
    return;
  }

  for (const file of files) {
    const frame = document.createElement("figure");
    frame.className = "w-72 shrink-0";
    const img = document.createElement("img");
    img.className = "aspect-video w-72 rounded border border-zinc-800 object-contain";
    img.loading = "lazy";
    img.alt = `${run.label} ${file}`;
    img.src = runUrl(run, file);
    const caption = document.createElement("figcaption");
    caption.className = "mt-2 text-center font-mono text-xs text-zinc-400";
    caption.textContent = `step ${stepFromName(file).toLocaleString()}`;
    frame.append(img, caption);
    host.appendChild(frame);
  }
}

function renderActiveRun(run) {
  const empty = document.getElementById("active-empty");
  const content = document.getElementById("active-content");
  if (!run) {
    empty.classList.remove("hidden");
    content.classList.add("hidden");
    return;
  }

  empty.classList.add("hidden");
  content.classList.remove("hidden");
  document.getElementById("active-title").textContent = run.label || run.name;
  document.getElementById("active-meta").textContent = run.cached ? `${run.name} (cached)` : run.name;
  document.getElementById("active-step").textContent = Number(run.latest_step || 0).toLocaleString();
  const latest = run.latest_metrics || {};
  document.getElementById("active-loss").textContent = fmtNumber(latest.loss_total ?? latest.loss, 5);

  const rows = run.loss_curve || [];
  lossChart.data.datasets = [
    { label: "loss_total", data: xy(rows, "loss_total", "loss"), borderColor: colors.total, backgroundColor: colors.total, pointRadius: 0, tension: 0 },
    { label: "loss_charbonnier", data: xy(rows, "loss_charbonnier", "l1", "t_l1"), borderColor: colors.charbonnier, backgroundColor: colors.charbonnier, pointRadius: 0, tension: 0 },
    { label: "loss_lpips", data: xy(rows, "loss_lpips", "t_lpips"), borderColor: colors.lpips, backgroundColor: colors.lpips, pointRadius: 0, tension: 0 },
  ].filter((dataset) => dataset.data.length);
  lossChart.update();

  const scoreRows = run.score_log || [];
  document.getElementById("score-meta").textContent = scoreRows.length
    ? `${scoreRows.length} held-out eval row${scoreRows.length === 1 ? "" : "s"}`
    : "No held-out eval rows published yet.";
  scoreChart.data.datasets = [
    { label: "model PSNR", yAxisID: "y", data: xy(scoreRows, "model_psnr_mean"), borderColor: colors.psnr, backgroundColor: colors.psnr, pointRadius: 2, tension: 0 },
    { label: "bicubic PSNR", yAxisID: "y", data: xy(scoreRows, "bicubic_psnr_mean"), borderColor: colors.bicubic, backgroundColor: colors.bicubic, borderDash: [4, 4], pointRadius: 2, tension: 0 },
    { label: "model LPIPS", yAxisID: "y1", data: xy(scoreRows, "model_lpips_mean"), borderColor: colors.lpips, backgroundColor: colors.lpips, pointRadius: 2, tension: 0 },
  ].filter((dataset) => dataset.data.length);
  scoreChart.update();

  renderVizStrip("viz-strip", "viz-count", run, 8);
}

function setupRunPicker(data) {
  const select = document.getElementById("run-select");
  select.replaceChildren();
  const activeRuns = (data.runs || []).filter((run) => run.active);
  const selectable = activeRuns.length ? activeRuns : (data.runs || []);
  for (const run of selectable) {
    const option = document.createElement("option");
    option.value = run.name;
    option.textContent = run.label || run.name;
    select.appendChild(option);
  }
  select.disabled = selectable.length === 0;
  select.addEventListener("change", () => {
    renderActiveRun(selectable.find((run) => run.name === select.value));
  });
  renderActiveRun(selectable[0] || null);
}

function renderMeasuredResult(data) {
  const v5 = (data.runs || []).find((run) => run.name === "srcnn-v5-pixel-temporal-validated");
  renderVizStrip("result-viz-strip", "result-viz-count", v5, 4);
}

async function loadDashboard() {
  const state = document.getElementById("data-state");
  try {
    const response = await fetch("data.json", { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    dashboardData = await response.json();
    state.textContent = "loaded";
    state.classList.remove("border-zinc-700");
    state.classList.add("border-emerald-700", "text-emerald-300");
    document.getElementById("updated-line").textContent = agoText(dashboardData.generated_at);
    setupRunPicker(dashboardData);
    renderMeasuredResult(dashboardData);
    window.setInterval(() => {
      document.getElementById("updated-line").textContent = agoText(dashboardData.generated_at);
    }, 30000);
  } catch (error) {
    state.textContent = "data load failed";
    state.classList.add("border-red-700", "text-red-300");
    document.getElementById("active-empty").classList.remove("hidden");
    document.getElementById("active-empty").textContent = `Could not load data.json: ${error.message}`;
  }
}

document.addEventListener("DOMContentLoaded", () => {
  setTheme(localStorage.getItem("oss-dashboard-theme") === "light");
  document.getElementById("theme-toggle").addEventListener("click", () => {
    setTheme(document.documentElement.classList.contains("dark"));
  });
  setupTabs();
  lossChart = makeLineChart("loss-chart", "loss");
  scoreChart = makeLineChart("score-chart", "score");
  scoreChart.options.scales.y1 = {
    type: "linear",
    position: "right",
    grid: { drawOnChartArea: false },
    ticks: { color: "#a1a1aa" },
    title: { display: true, text: "LPIPS", color: "#a1a1aa" },
  };
  loadDashboard();
});
</script>
</body>
</html>
"""


def main() -> None:
    data = build_data()
    write_data(data)
    write_index(extract_pitch())
    print(f"wrote {DATA_JSON.relative_to(ROOT)} and {INDEX_HTML.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
