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

  <div class="panel">
    <h2>PSNR vs bicubic over training</h2>
    <canvas id="chart-psnr"></canvas>
  </div>

  <div class="panel">
    <h2>LPIPS vs bicubic ↓ better</h2>
    <canvas id="chart-lpips"></canvas>
  </div>

  <div class="panel">
    <h2>Loss</h2>
    <canvas id="chart-loss"></canvas>
  </div>

  <div class="panel">
    <h2>SSIM</h2>
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
    options: {
      animation: false,
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: 'nearest', axis: 'x', intersect: false },
      plugins: { legend: { labels: { color: '#e6edf3' } } },
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

function buildCharts() {
  charts.psnr = lineChart('chart-psnr', 'PSNR (dB)', null, { yLabel: 'PSNR (dB)' });
  charts.lpips = lineChart('chart-lpips', 'LPIPS', null, { yLabel: 'LPIPS (lower=better)', yMin: 0 });
  charts.loss = lineChart('chart-loss', 'loss');
  charts.ssim = lineChart('chart-ssim', 'SSIM', null, { yLabel: 'SSIM', yMin: 0, yMax: 1 });
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

async function refresh() {
  try {
    const [info, metrics, score, logTail] = await Promise.all([
      fetchJSON('/api/info'),
      fetchJSON('/api/metrics'),
      fetchJSON('/api/score'),
      fetchText('/api/log'),
    ]);

    lastUpdate = Date.now();

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
    const trainXY = (key) =>
      train.map(r => ({ x: r.step, y: r[key] })).filter(p => p.y !== undefined && p.y !== null);

    setChart(charts.loss, [
      { label: 'loss', data: trainXY('loss'), borderColor: '#58a6ff', backgroundColor: '#58a6ff', tension: 0, pointRadius: 0 },
      { label: 'l1', data: trainXY('l1'), borderColor: '#d29922', backgroundColor: '#d29922', tension: 0, pointRadius: 0 },
    ]);

    setChart(charts.ssim, [
      { label: 'SSIM (train)', data: trainXY('ssim'), borderColor: '#3fb950', backgroundColor: '#3fb950', tension: 0, pointRadius: 0 },
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
    setChart(charts.psnr, [
      { label: 'model', data: evalXY('model_psnr_mean'), borderColor: '#3fb950', backgroundColor: '#3fb950', tension: 0, pointRadius: 3 },
      { label: 'bicubic', data: evalXY('bicubic_psnr_mean'), borderColor: '#8b949e', backgroundColor: '#8b949e', tension: 0, pointRadius: 3, borderDash: [4, 4] },
    ]);
    setChart(charts.lpips, [
      { label: 'model', data: evalXY('model_lpips_mean'), borderColor: '#3fb950', backgroundColor: '#3fb950', tension: 0, pointRadius: 3 },
      { label: 'bicubic', data: evalXY('bicubic_lpips_mean'), borderColor: '#8b949e', backgroundColor: '#8b949e', tension: 0, pointRadius: 3, borderDash: [4, 4] },
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
                self._send_json(self._read_json("metrics.json", default=[]))
            elif path == "/api/score":
                self._send_json(self._read_json("score_log.json", default=[]))
            elif path == "/api/log":
                self._send_text(self._read_log_tail(n_lines=200))
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
