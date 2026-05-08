const colors = {
  total: "#6ee7b7",
  charbonnier: "#a8d4a0",
  lpips: "#c8a960",
  msvgg: "#d6f4cc",
  sobel: "#7dd3fc",
  tc: "#86efac",
  wavelet: "#fde68a",
  ganG: "#f0b56f",
  v5kd: "#d8b4fe",
  psnr: "#d6f4cc",
  bicubic: "#4f6a64",
  gold: "#c8a960",
  v4: "#f0b56f",
  v5: "#fbbf24",
  nvidia: "#8cffc1",
  amd: "#ff7b7b",
  intel: "#7dd3fc",
  ossPoint: "#f4f4f5",
};

const lossComponentDefs = [
  { key: "loss_charbonnier", label: "loss_charbonnier", color: colors.charbonnier, aliases: ["l1", "t_l1"] },
  { key: "loss_lpips", label: "loss_lpips", color: colors.lpips, aliases: ["lpips", "t_lpips"] },
  { key: "loss_msvgg", label: "loss_msvgg", color: colors.msvgg, aliases: [] },
  { key: "loss_sobel", label: "loss_sobel", color: colors.sobel, aliases: [] },
  { key: "loss_tc", label: "loss_tc", color: colors.tc, aliases: [] },
  { key: "loss_wavelet", label: "loss_wavelet", color: colors.wavelet, aliases: [] },
  { key: "loss_gan_g", label: "loss_gan_g", color: colors.ganG, aliases: [] },
  { key: "loss_v5_kd", label: "loss_v5_kd", color: colors.v5kd, aliases: [] },
];

const comparisonReferenceLines = [
  { id: "bicubic", method: "Bicubic baseline", latest: true, psnr: 23.909, lpips: 0.2945, color: colors.bicubic, source: "measured directly on the OSS TartanAir oldtown held-out batch", nudge: -12 },
  { id: "fsr1", method: "FSR 1", latest: false, psnr: 26.0, lpips: 0.22, color: "#ffb3a8", source: "AMD GPUOpen FSR1 spatial-upscaler docs plus Digital Foundry comparisons" },
  { id: "fsr2", method: "FSR 2", latest: false, psnr: 28.0, lpips: 0.18, color: "#ff8f8f", source: "AMD FSR2 performance-mode material plus Digital Foundry / Insider Gaming comparisons" },
  { id: "fsr3", method: "FSR 3", latest: true, psnr: 28.5, lpips: 0.17, color: colors.amd, source: "AMD-published FSR3 material; SR component only, frame generation not directly comparable", nudge: -8 },
  { id: "xess1", method: "XeSS 1.x", latest: true, psnr: 28.5, lpips: 0.17, color: colors.intel, source: "Intel XeSS paper / developer guidance and Cyberpunk 2077 / Hitman 3 benchmark coverage", nudge: 10 },
  { id: "dlss1", method: "DLSS 1", latest: false, psnr: 25.0, lpips: 0.22, color: "#c7ffd8", source: "NVIDIA archived DLSS material plus Digital Foundry comparisons of the per-game CNN era" },
  { id: "dlss2", method: "DLSS 2", latest: false, psnr: 30.0, lpips: 0.14, color: "#a7ffc9", source: "NVIDIA DLSS 2 publications and Digital Foundry quality-mode coverage" },
  { id: "dlss3", method: "DLSS 3", latest: false, psnr: 30.5, lpips: 0.13, color: "#92ffc1", source: "NVIDIA DLSS 3 publications; SR component only, frame generation not directly comparable" },
  { id: "dlss4", method: "DLSS 4", latest: true, psnr: 32.5, lpips: 0.10, color: colors.nvidia, source: "NVIDIA DLSS 4 transformer publications plus Digital Foundry / TechPowerUp image-quality coverage", nudge: 0 },
];

const v4FallbackCrossVersionPoints = [
  { name: "v4 (SRGD)", label_full: "v4 single-frame on SRGD held-out (in-distribution)", psnr: 33.67, lpips: 0.270, step: 300000, manifest: "SRGD held-out, 8 frames", is_in_distribution: true },
  { name: "v4 (TartanAir, distribution-shifted)", label_full: "v4 single-frame on TartanAir oldtown (NOT in v4's training distribution)", psnr: 11.718, lpips: 0.6367, step: 300000, manifest: "TartanAir oldtown, 64 frames (same as v5/v6.1)", is_in_distribution: false },
];

const badDataSeriesIds = new Set(["v4-srgd", "v4-tartanair"]);
const comparisonMetricScale = {
  psnr: { min: 0, max: 35, pad: 0.5, precision: 1 },
  lpips: { min: 0, max: 1, pad: 0.03, precision: 100 },
};

const chartInputEvents = ["mousemove", "mouseout", "click", "mousedown", "mouseup", "wheel", "touchstart", "touchmove", "touchend"];

function fmtNumber(value, digits = 4) {
  if (value === undefined || value === null || Number.isNaN(Number(value))) return "--";
  const num = Number(value);
  if (Math.abs(num) >= 1000) return num.toLocaleString();
  return num.toFixed(digits).replace(/\.?0+$/, "");
}

function formatStepTick(value) {
  const step = Number(value);
  if (!Number.isFinite(step)) return "";
  if (step >= 1000000) return `${(step / 1000000).toFixed(1).replace(/\.0$/, "")}M`;
  if (step >= 1000) {
    const thousands = step / 1000;
    return `${thousands >= 100 ? Math.round(thousands).toLocaleString() : thousands.toFixed(1).replace(/\.0$/, "")}K`;
  }
  return Math.round(step).toLocaleString();
}

function valueFromRow(row, key, aliases = []) {
  let value = row?.[key];
  for (const alias of aliases) {
    if (value !== undefined && value !== null) break;
    value = row?.[alias];
  }
  return value;
}

function xy(rows, key, ...aliases) {
  return (rows || []).map((row) => {
    const value = valueFromRow(row, key, aliases);
    if (value === undefined || value === null) return null;
    return { x: Number(row.step || 0), y: Number(value) };
  }).filter(Boolean);
}

function xyPositive(rows, key, aliases = []) {
  return (rows || []).map((row) => {
    const value = Number(valueFromRow(row, key, aliases));
    return { x: Number(row.step || 0), y: Number.isFinite(value) && value > 0 ? value : null };
  });
}

function hasMetricKey(rows, key, aliases = []) {
  return (rows || []).some((row) => valueFromRow(row, key, aliases) !== undefined && valueFromRow(row, key, aliases) !== null);
}

function trainingPsnrValue(row) {
  return row?.psnr_db ?? row?.psnr ?? row?.t_psnr ?? row?.psnr_proxy;
}

function trainingPsnrUsesProxy(rows) {
  return (rows || []).some((row) => {
    const real = row?.psnr_db ?? row?.psnr ?? row?.t_psnr;
    return (real === undefined || real === null) && row?.psnr_proxy !== undefined && row?.psnr_proxy !== null;
  });
}

function runTokenFromName(name) {
  return String(name || "").replace(/^srcnn-/, "");
}

function runAliases(run) {
  const token = runTokenFromName(run?.name);
  const aliases = new Set([token]);
  if (token) {
    aliases.add(token.replace(/-pico-\d+$/, ""));
    aliases.add(token.replace(/-pixel-temporal-validated$/, ""));
    aliases.add(token.replace(/^prod-/, ""));
  }
  return Array.from(aliases).filter(Boolean);
}

function findRunByUrlToken(data, token) {
  const query = String(token || "").trim();
  if (!query) return null;
  return (data?.runs || []).find((run) =>
    runAliases(run).some((alias) => alias === query || alias.startsWith(`${query}-`))
  ) || null;
}

function activeRunFrom(data) {
  return (data?.runs || []).find((run) => run.active) || (data?.runs || [])[0] || null;
}

function runForOptions(data, opts) {
  return findRunByUrlToken(data, opts.run) || activeRunFrom(data);
}

function chartZoomOptions(opts = {}) {
  const enabled = !opts.embed;
  return {
    pan: { enabled, mode: "xy", modifierKey: "shift" },
    zoom: {
      wheel: { enabled, modifierKey: null },
      pinch: { enabled },
      drag: { enabled, backgroundColor: "rgba(56, 189, 248, 0.15)", borderColor: "rgba(56, 189, 248, 0.6)", borderWidth: 1, threshold: 6 },
      mode: "xy",
    },
    limits: {
      x: { min: "original", max: "original" },
      y: { min: "original", max: "original" },
      y1: { min: "original", max: "original" },
    },
  };
}

function chartOptions({ yTitle = "loss", componentAxis = false, logY = false } = {}, opts = {}) {
  const scales = {
    x: {
      type: "linear",
      title: { display: true, text: "step", color: "#a1a1aa" },
      grid: { color: "rgba(113,113,122,0.22)" },
      ticks: { color: "#a1a1aa", maxTicksLimit: 6, autoSkip: true, callback: formatStepTick },
    },
    y: {
      type: logY ? "logarithmic" : "linear",
      title: { display: true, text: yTitle, color: "#a1a1aa" },
      grid: { color: "rgba(113,113,122,0.22)" },
      ticks: { color: "#a1a1aa", maxTicksLimit: 6 },
    },
  };
  if (componentAxis) {
    scales.y1 = {
      type: "linear",
      position: "right",
      grid: { drawOnChartArea: false },
      ticks: { color: "#a1a1aa", maxTicksLimit: 6 },
      title: { display: true, text: "component loss", color: "#a1a1aa" },
    };
  }
  return {
    responsive: true,
    maintainAspectRatio: false,
    animation: false,
    parsing: false,
    normalized: true,
    events: opts.embed ? ["mousemove", "mouseout", "click"] : chartInputEvents,
    interaction: { mode: "point", intersect: true },
    scales,
    plugins: {
      legend: { labels: { color: "#d4d4d8", boxWidth: 12 } },
      tooltip: { mode: "point", intersect: true },
      zoom: chartZoomOptions(opts),
    },
  };
}

function applyEmbedCrowdingRules(options, chartId, opts = {}) {
  if (!opts.embed) return options;
  options.maintainAspectRatio = false;
  if (chartId === "psnr-held-out" || chartId === "psnr-inflight") {
    options.plugins.legend.display = false;
  }
  return options;
}

function referenceLine(rows, value) {
  const steps = (rows || []).map((row) => Number(row.step)).filter(Number.isFinite);
  if (!steps.length || !Number.isFinite(Number(value))) return [];
  return [{ x: Math.min(...steps), y: Number(value) }, { x: Math.max(...steps), y: Number(value) }];
}

function chartTooltipLines(row, run) {
  const lines = [
    `loss_total: ${fmtNumber(row.loss_total ?? row.loss, 5)}`,
    ...lossComponentDefs.map((def) => `${def.label}: ${fmtNumber(valueFromRow(row, def.key, def.aliases), 5)}`),
  ];
  if (row.gpu_status) lines.push(`gpu_status: ${row.gpu_status}`);
  else if (run.gpu_status?.captured_at && Number(row.step || 0) === Number(run.latest_step || 0)) {
    lines.push(`gpu_status: ${run.gpu_status.utilization_pct}% util, ${run.gpu_status.memory_used_mib}/${run.gpu_status.memory_total_mib} MiB`);
  }
  return lines;
}

function createChart(canvasEl, config) {
  const ChartCtor = window.Chart;
  if (!ChartCtor) throw new Error("Chart.js is not loaded");
  return new ChartCtor(canvasEl, config);
}

function buildLossCurve(canvasEl, data, opts = {}) {
  const run = runForOptions(data, opts);
  if (!run) throw new Error("run not found");
  const rows = run.loss_curve || [];
  const datasets = [
    { label: "loss_total", yAxisID: "y", data: xy(rows, "loss_total", "loss"), borderColor: colors.total, backgroundColor: colors.total, pointRadius: 0, pointHitRadius: 6, pointHoverRadius: 5, borderWidth: 2, tension: 0 },
    { label: "loss_charbonnier", yAxisID: "y1", data: xy(rows, "loss_charbonnier", "l1", "t_l1"), borderColor: colors.charbonnier, backgroundColor: colors.charbonnier, pointRadius: 0, pointHitRadius: 6, borderWidth: 1.5, borderDash: [5, 3], tension: 0 },
    { label: "loss_lpips", yAxisID: "y1", data: xy(rows, "loss_lpips", "t_lpips"), borderColor: colors.lpips, backgroundColor: colors.lpips, pointRadius: 0, pointHitRadius: 6, borderWidth: 1.5, tension: 0 },
  ].filter((dataset) => dataset.data.length);
  if (!datasets.length) throw new Error("no loss curve points for run");
  const options = chartOptions({ yTitle: "total loss", componentAxis: true }, opts);
  options.plugins.tooltip = {
    mode: "point",
    intersect: true,
    callbacks: {
      title(items) {
        const row = rows[items[0]?.dataIndex] || {};
        return `step ${Number(row.step || items[0]?.parsed?.x || 0).toLocaleString()}`;
      },
      label() { return ""; },
      afterBody(items) {
        const row = rows[items[0]?.dataIndex] || {};
        return chartTooltipLines(row, run);
      },
    },
  };
  return createChart(canvasEl, { type: "line", data: { datasets }, options: applyEmbedCrowdingRules(options, "loss-curve", opts) });
}

function buildLossDecomposition(canvasEl, data, opts = {}) {
  const run = runForOptions(data, opts);
  if (!run) throw new Error("run not found");
  const rows = run.loss_curve || [];
  const datasets = lossComponentDefs.filter((def) => hasMetricKey(rows, def.key, def.aliases)).map((def) => ({
    label: def.label,
    data: xyPositive(rows, def.key, def.aliases),
    borderColor: def.color,
    backgroundColor: def.color,
    pointRadius: 0,
    pointHitRadius: 6,
    pointHoverRadius: 4,
    borderWidth: 1.8,
    spanGaps: true,
    tension: 0,
  }));
  if (!datasets.length) throw new Error("no loss decomposition points for run");
  const options = chartOptions({ yTitle: "loss component value (log scale)", logY: true }, opts);
  return createChart(canvasEl, { type: "line", data: { datasets }, options: applyEmbedCrowdingRules(options, "loss-decomp", opts) });
}

function buildPsnrInflight(canvasEl, data, opts = {}) {
  const run = runForOptions(data, opts);
  if (!run) throw new Error("run not found");
  const rows = run.loss_curve || [];
  const series = rows.map((row) => {
    const value = trainingPsnrValue(row);
    if (value === undefined || value === null) return null;
    return { x: Number(row.step || 0), y: Number(value) };
  }).filter(Boolean);
  if (!series.length) throw new Error("no in-flight PSNR points for run");
  const ref = referenceLine(rows, 25.703);
  const usesProxy = trainingPsnrUsesProxy(rows);
  const datasets = [
    { label: `${run.history?.title || run.label || "run"} ${usesProxy ? "PSNR proxy" : "PSNR"}`, data: series, borderColor: colors.psnr, backgroundColor: colors.psnr, pointRadius: 0, pointHitRadius: 6, pointHoverRadius: 4, borderWidth: 2, tension: 0 },
    { label: "v5 held-out PSNR ref 25.703 dB", data: ref, borderColor: colors.gold, backgroundColor: colors.gold, pointRadius: 0, pointHitRadius: 6, borderWidth: 1.5, borderDash: [5, 4], tension: 0 },
  ].filter((dataset) => dataset.data.length);
  const options = chartOptions({ yTitle: "training-crop PSNR dB" }, opts);
  return createChart(canvasEl, { type: "line", data: { datasets }, options: applyEmbedCrowdingRules(options, "psnr-inflight", opts) });
}

function buildLpipsInflight(canvasEl, data, opts = {}) {
  const run = runForOptions(data, opts);
  if (!run) throw new Error("run not found");
  const rows = run.loss_curve || [];
  const series = xy(rows, "loss_lpips", "lpips", "t_lpips");
  if (!series.length) throw new Error("no in-flight LPIPS points for run");
  const ref = referenceLine(rows, 0.1666);
  const options = chartOptions({ yTitle: "training-crop LPIPS" }, opts);
  options.scales.y.min = 0;
  options.scales.y.max = 1;
  const datasets = [
    { label: `${run.history?.title || run.label || "run"} LPIPS`, data: series, borderColor: colors.lpips, backgroundColor: colors.lpips, pointRadius: 0, pointHitRadius: 6, pointHoverRadius: 4, borderWidth: 2, tension: 0 },
    { label: "v5 held-out LPIPS ref 0.1666", data: ref, borderColor: colors.gold, backgroundColor: colors.gold, pointRadius: 0, pointHitRadius: 6, borderWidth: 1.5, borderDash: [5, 4], tension: 0 },
  ].filter((dataset) => dataset.data.length);
  return createChart(canvasEl, { type: "line", data: { datasets }, options: applyEmbedCrowdingRules(options, "lpips-inflight", opts) });
}

function buildPsnrHeldOut(canvasEl, data, opts = {}) {
  const run = runForOptions(data, opts);
  if (!run) throw new Error("run not found");
  const rows = run.score_log || [];
  const modelPsnr = xy(rows, "model_psnr_mean");
  const bicubicPsnr = xy(rows, "bicubic_psnr_mean");
  const modelLpips = xy(rows, "model_lpips_mean");
  if (!modelPsnr.length && !bicubicPsnr.length && !modelLpips.length) throw new Error("no held-out score points for run");
  const options = chartOptions({ yTitle: "held-out PSNR dB" }, opts);
  options.scales.y1 = {
    type: "linear",
    position: "right",
    min: 0,
    max: 1,
    grid: { drawOnChartArea: false },
    ticks: { color: "#a1a1aa", maxTicksLimit: 6 },
    title: { display: true, text: "held-out LPIPS", color: "#a1a1aa" },
  };
  const datasets = [
    { label: "model PSNR", yAxisID: "y", data: modelPsnr, borderColor: colors.psnr, backgroundColor: colors.psnr, pointRadius: 3, pointHitRadius: 6, borderWidth: 2, tension: 0 },
    { label: "bicubic PSNR", yAxisID: "y", data: bicubicPsnr, borderColor: colors.bicubic, backgroundColor: colors.bicubic, pointRadius: 3, pointHitRadius: 6, borderWidth: 1.7, borderDash: [5, 4], tension: 0 },
    { label: "model LPIPS", yAxisID: "y1", data: modelLpips, borderColor: colors.lpips, backgroundColor: colors.lpips, pointRadius: 3, pointHitRadius: 6, borderWidth: 2, tension: 0 },
  ].filter((dataset) => dataset.data.length);
  return createChart(canvasEl, { type: "line", data: { datasets }, options: applyEmbedCrowdingRules(options, "psnr-held-out", opts) });
}

function normalizedComparisonValue(value) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : null;
}

function comparisonValuesFromSeries(series, metric) {
  return (series?.points || []).map((point) => normalizedComparisonValue(point?.[metric])).filter((value) => value !== null);
}

function roundedComparisonMin(value, metric) {
  const scale = comparisonMetricScale[metric];
  return metric === "psnr" ? Math.floor(value) : Math.floor(value * scale.precision) / scale.precision;
}

function roundedComparisonMax(value, metric) {
  const scale = comparisonMetricScale[metric];
  return metric === "psnr" ? Math.ceil(value) : Math.ceil(value * scale.precision) / scale.precision;
}

function clampComparisonBounds(bounds, metric) {
  const scale = comparisonMetricScale[metric];
  return { yMin: Math.max(scale.min, bounds.yMin), yMax: Math.min(scale.max, bounds.yMax) };
}

function comparisonScorePoint(row, run, options = {}) {
  if (!row) return null;
  const step = Number(row.step ?? options.step ?? run?.latest_step ?? 0);
  const psnr = Number(row.model_psnr_mean);
  const lpips = Number(row.model_lpips_mean);
  if (!Number.isFinite(step) || !Number.isFinite(psnr) || !Number.isFinite(lpips)) return null;
  const runLabel = run.history?.title || run.label || run.name;
  return {
    step,
    psnr,
    lpips,
    active: Boolean(run.active),
    run: runLabel,
    name: options.name || runLabel,
    label_full: options.label_full || runLabel,
    manifest: options.manifest || row.manifest || `held-out batch, ${row.n_samples || "?"} frames`,
    color: options.color || colors.ossPoint,
    is_in_distribution: options.is_in_distribution ?? true,
    hollow: Boolean(options.hollow),
  };
}

function comparisonPointsFromScoreLog(run, options = {}) {
  return (run?.score_log || []).map((row) => comparisonScorePoint(row, run, options)).filter(Boolean).sort((a, b) => a.step - b.step);
}

function v4CrossVersionPoints(run) {
  const source = (run?.cross_version_points || []).length ? run.cross_version_points : v4FallbackCrossVersionPoints;
  return source.map((point) => ({ ...point, step: Number(point.step || 300000), run: point.name, color: colors.v4 }))
    .filter((point) => Number.isFinite(Number(point.psnr)) && Number.isFinite(Number(point.lpips)));
}

function comparisonTrajectorySeries(data) {
  const runs = data?.runs || [];
  const byName = new Map(runs.map((run) => [run.name, run]));
  const v4Run = byName.get("srcnn-prod-v4-lpips") || {};
  const v4Cross = v4CrossVersionPoints(v4Run);
  const v4SrgdFallback = v4Cross.find((point) => point.is_in_distribution !== false);
  const v4Srgd = comparisonPointsFromScoreLog(v4Run, {
    name: "v4-SRGD",
    label_full: "v4-SRGD held-out trajectory (in-distribution)",
    manifest: v4SrgdFallback?.manifest || "SRGD held-out, 8 frames",
    color: colors.v4,
    is_in_distribution: true,
  });
  const v4Tartan = v4Cross.find((point) => point.is_in_distribution === false);
  const v5 = comparisonPointsFromScoreLog(byName.get("srcnn-v5-pixel-temporal-validated"), {
    name: "v5",
    label_full: "v5-pixel-temporal on TartanAir oldtown",
    manifest: "TartanAir oldtown, 64 frames",
    color: colors.v5,
  });
  const v61Run = byName.get("srcnn-v6.1-pico-001");
  const v61 = comparisonPointsFromScoreLog(v61Run, {
    name: "v6.1",
    label_full: "v6.1 Pico on TartanAir oldtown",
    manifest: "TartanAir oldtown, 64 frames",
    color: colors.psnr,
  });
  const series = [
    { id: "v4-srgd", label: "v4-SRGD", color: colors.v4, points: v4Srgd.length ? v4Srgd : (v4SrgdFallback ? [{ ...v4SrgdFallback, name: "v4-SRGD", label_full: "v4-SRGD held-out (in-distribution)" }] : []) },
    { id: "v4-tartanair", label: "v4-TartanAir", color: colors.v4, hollow: true, points: v4Tartan ? [{ ...v4Tartan, name: "v4-TartanAir", hollow: true }] : [] },
    { id: "v5", label: "v5", color: colors.v5, points: v5 },
    { id: "v61", label: "v6.1", color: colors.psnr, active: Boolean(v61Run?.active), points: v61 },
  ].filter((item) => item.points.length).filter((item) => !badDataSeriesIds.has(item.id));
  if (v61Run?.active && v61.length) v61[v61.length - 1].live = true;
  if (v5.length === 1) {
    v5[0].markerLabel = `v5 final · ${formatStepTick(v5[0].step)} steps`;
    v5[0].markerLabelColor = colors.v5;
  }
  const allSteps = series.flatMap((item) => item.points.map((point) => point.step)).filter(Number.isFinite);
  const maxStep = Math.max(100000, ...allSteps);
  return { series, xMin: 0, xMax: Math.ceil(maxStep / 5000) * 5000, logMin: 1 };
}

function comparisonDatasetForSeries(series, metric) {
  return {
    label: series.label,
    yAxisID: "y",
    data: series.points.map((point) => ({ x: point.step, y: point[metric], point })),
    borderColor: series.color,
    backgroundColor: series.color,
    pointBorderColor: series.color,
    pointBackgroundColor(context) {
      return context.raw?.point?.hollow || series.hollow ? "#06110f" : series.color;
    },
    pointRadius: series.id === "v4-srgd" ? 4 : (series.points.length === 1 ? 8 : 6),
    pointHitRadius: 6,
    pointHoverRadius: series.id === "v4-srgd" ? 6 : (series.points.length === 1 ? 10 : 8),
    pointBorderWidth(context) {
      return series.points.length === 1 || context.raw?.point?.hollow || series.hollow ? 3 : 2;
    },
    showLine: series.points.length > 1,
    borderWidth: 1.8,
    tension: 0,
    runLine: true,
  };
}

function lineSpan(value, yAxisID, label, color, bounds, options = {}) {
  return {
    label,
    yAxisID,
    data: [{ x: bounds.xMin, y: value }, { x: bounds.xMax, y: value }],
    borderColor: color,
    backgroundColor: color,
    borderWidth: options.solid ? 1.5 : 1,
    borderDash: options.solid ? [] : [5, 4],
    pointRadius: 0,
    pointHitRadius: 6,
    pointHoverRadius: 0,
    tension: 0,
    refLine: true,
    refName: options.refName || label,
    refSource: options.source || "",
    refNudge: options.nudge || 0,
    hidden: Boolean(options.hidden),
  };
}

const crossVersionLabelPlugin = {
  id: "embedCrossVersionLabels",
  afterDatasetsDraw(chart) {
    const { ctx, chartArea, scales } = chart;
    if (!chartArea || !scales?.x) return;
    ctx.save();
    ctx.font = "10px ui-monospace, SFMono-Regular, Menlo, monospace";
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    chart.data.datasets.forEach((dataset, index) => {
      if (!dataset.refLine || !chart.isDatasetVisible(index)) return;
      const scale = scales[dataset.yAxisID || "y"];
      const value = dataset.data?.[0]?.y;
      if (!scale || !Number.isFinite(Number(value))) return;
      const y = scale.getPixelForValue(value) + Number(dataset.refNudge || 0);
      if (y < chartArea.top + 6 || y > chartArea.bottom - 6) return;
      ctx.fillStyle = dataset.borderColor;
      ctx.globalAlpha = 0.82;
      ctx.fillText(dataset.refName, chartArea.right - 4, y);
      ctx.globalAlpha = 1;
    });
    ctx.restore();
  },
};

function buildCrossVersion(canvasEl, data, opts = {}, metric = "psnr") {
  const trajectories = comparisonTrajectorySeries(data);
  const isPsnr = metric === "psnr";
  const visibleRefs = comparisonReferenceLines.filter((ref) => ref.latest);
  const visibleValues = [
    ...trajectories.series.flatMap((item) => comparisonValuesFromSeries(item, metric)),
    ...visibleRefs.map((ref) => normalizedComparisonValue(ref[metric])).filter((value) => value !== null),
  ];
  if (!visibleValues.length) throw new Error("no cross-version comparison points");
  const scale = comparisonMetricScale[metric];
  const bounds = clampComparisonBounds({
    yMin: roundedComparisonMin(Math.min(...visibleValues) - scale.pad, metric),
    yMax: roundedComparisonMax(Math.max(...visibleValues) + scale.pad, metric),
  }, metric);
  const datasets = [
    ...trajectories.series.map((series) => comparisonDatasetForSeries(series, metric)),
    ...comparisonReferenceLines.map((ref) => {
      const hidden = !ref.latest;
      const refNudge = ref.id === "bicubic" ? (isPsnr ? -18 : 14) : ref.nudge;
      return lineSpan(ref[metric], "y", `${ref.method} ${isPsnr ? "PSNR" : "LPIPS"} ref`, ref.color, trajectories, {
        hidden,
        solid: ref.id === "bicubic",
        refName: `${ref.method} ${isPsnr ? "PSNR" : "LPIPS"}`,
        source: ref.source,
        nudge: refNudge,
      });
    }),
  ];
  const options = {
    responsive: true,
    maintainAspectRatio: false,
    animation: false,
    parsing: false,
    normalized: true,
    events: opts.embed ? ["mousemove", "mouseout", "click"] : chartInputEvents,
    interaction: { mode: "point", intersect: true },
    scales: {
      x: {
        type: "linear",
        min: trajectories.xMin,
        max: trajectories.xMax,
        title: { display: true, text: "Training step", color: "#a1a1aa" },
        grid: { color: "rgba(113,113,122,0.18)" },
        ticks: { color: "#a1a1aa", autoSkip: true, maxRotation: 0, maxTicksLimit: 7, callback: formatStepTick },
      },
      y: {
        min: bounds.yMin,
        max: bounds.yMax,
        reverse: !isPsnr,
        title: { display: true, text: isPsnr ? "PSNR dB ↑" : "LPIPS ↓", color: "#a1a1aa" },
        grid: { color: "rgba(113,113,122,0.22)" },
        ticks: { color: "#a1a1aa", maxTicksLimit: 7 },
      },
    },
    plugins: {
      legend: { labels: { color: "#d4d4d8", boxWidth: 10 } },
      tooltip: {
        mode: "point",
        intersect: true,
        filter(item) {
          return Boolean(item.dataset?.refLine || item.raw?.point);
        },
        callbacks: {
          title(items) {
            const raw = items[0]?.raw;
            if (raw?.point) return raw.point.label_full || raw.point.run;
            return items[0]?.dataset?.refName || items[0]?.dataset?.label || "";
          },
          label(item) {
            const dataset = item.dataset || {};
            if (dataset.refLine) return `${isPsnr ? "PSNR" : "LPIPS"}: ${fmtNumber(item.parsed.y, isPsnr ? 2 : 4)} - ${dataset.refSource}`;
            const point = item.raw?.point;
            if (point) {
              return [
                `${isPsnr ? "PSNR" : "LPIPS"}: ${fmtNumber(item.parsed.y, isPsnr ? 3 : 4)}`,
                `Run: ${point.name || point.run || item.dataset.label}`,
                `Held-out batch: ${point.manifest || "unknown"}`,
                `Step: ${Number(point.step || 0).toLocaleString()}`,
              ];
            }
            return `${isPsnr ? "PSNR" : "LPIPS"}: ${fmtNumber(item.parsed.y, isPsnr ? 3 : 4)}`;
          },
        },
      },
      zoom: chartZoomOptions(opts),
    },
  };
  return createChart(canvasEl, { type: "line", data: { datasets }, options, plugins: [crossVersionLabelPlugin] });
}

export function buildChartById(chartId, canvasEl, data, opts = {}) {
  const id = String(chartId || "").trim();
  switch (id) {
    case "loss":
    case "loss-curve":
      return buildLossCurve(canvasEl, data, opts);
    case "loss-decomp":
    case "loss-decomposition":
      return buildLossDecomposition(canvasEl, data, opts);
    case "psnr-live":
    case "psnr-inflight":
      return buildPsnrInflight(canvasEl, data, opts);
    case "lpips-live":
    case "lpips-inflight":
      return buildLpipsInflight(canvasEl, data, opts);
    case "psnr-held-out":
    case "lpips-held-out":
      return buildPsnrHeldOut(canvasEl, data, opts);
    case "score-progression":
    case "cross-version-aggregate":
    case "cross-version-psnr":
      return buildCrossVersion(canvasEl, data, opts, "psnr");
    case "cross-version-lpips":
      return buildCrossVersion(canvasEl, data, opts, "lpips");
    default:
      throw new Error(`unknown chart id: ${id}`);
  }
}
