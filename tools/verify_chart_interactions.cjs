#!/usr/bin/env node
"use strict";

const fs = require("node:fs");
const path = require("node:path");
const { execSync } = require("node:child_process");

const ROOT = process.env.OSS_ROOT || "/Users/cashconway/OpenSuperSampling";
const URL = process.env.OSS_URL || "https://opensupersampling.org";
const CHART_SELECTOR = process.env.OSS_CHART_SELECTOR || "#cross-version-psnr-chart";
const HEADLESS = process.env.HEADLESS !== "0";
const VIEWPORT = { width: 1440, height: 900 };
const SCREENSHOTS = {
  crtOff: "oss-chart-crt-off-desktop-1440x900.png",
  crtOn: "oss-chart-crt-on-desktop-1440x900.png",
  fullscreen: "oss-chart-fullscreen-desktop-1440x900.png",
  interactions: "oss-chart-interactions-desktop-1440x900.png",
};

function requirePlaywright() {
  const candidates = [];
  if (process.env.PLAYWRIGHT_MODULE) candidates.push(process.env.PLAYWRIGHT_MODULE);
  try {
    candidates.push(require.resolve("playwright"));
  } catch {}
  try {
    candidates.push(path.join(execSync("npm root -g", { encoding: "utf8" }).trim(), "playwright"));
  } catch {}

  for (const candidate of candidates) {
    try {
      return require(candidate);
    } catch {}
  }
  throw new Error(
    "Could not load Playwright. Install it globally (`npm i -g playwright && playwright install chromium`) " +
    "or set PLAYWRIGHT_MODULE=/path/to/node_modules/playwright."
  );
}

function assert(condition, message, details = undefined) {
  if (!condition) {
    const suffix = details === undefined ? "" : `\n${JSON.stringify(details, null, 2)}`;
    throw new Error(`${message}${suffix}`);
  }
}

function scaleRange(scale) {
  return Math.abs(Number(scale.max) - Number(scale.min));
}

function scaleChanged(before, after, axis) {
  const baseRange = Math.max(scaleRange(before[axis]), 1);
  const tol = baseRange * 0.002;
  return Math.abs(after[axis].min - before[axis].min) > tol ||
    Math.abs(after[axis].max - before[axis].max) > tol;
}

function scaleUnchanged(before, after, axis) {
  return !scaleChanged(before, after, axis);
}

async function chartState(page, selector = CHART_SELECTOR) {
  return page.evaluate((sel) => {
    const canvas = document.querySelector(sel);
    if (!canvas) return null;
    const chart = window.Chart?.getChart?.(canvas);
    if (!chart) return null;
    const scales = {};
    for (const [id, scale] of Object.entries(chart.scales || {})) {
      scales[id] = { min: Number(scale.min), max: Number(scale.max) };
    }
    const rect = canvas.getBoundingClientRect();
    const area = chart.chartArea;
    return {
      selector: sel,
      canvasId: canvas.id || null,
      rect: { left: rect.left, top: rect.top, width: rect.width, height: rect.height },
      chartArea: { left: area.left, right: area.right, top: area.top, bottom: area.bottom },
      x: scales.x,
      y: scales.y,
      y1: scales.y1 || null,
    };
  }, selector);
}

async function resetZoom(page, selector = CHART_SELECTOR) {
  await page.evaluate((sel) => {
    const canvas = document.querySelector(sel);
    const chart = canvas ? window.Chart?.getChart?.(canvas) : null;
    chart?.resetZoom?.("none");
    chart?.update?.("none");
  }, selector);
  await page.waitForTimeout(150);
}

async function drag(page, from, to, steps = 16) {
  await page.mouse.move(from.x, from.y);
  await page.mouse.down();
  await page.mouse.move(to.x, to.y, { steps });
  await page.mouse.up();
  await page.waitForTimeout(250);
}

function pointsFor(state) {
  const r = state.rect;
  const a = state.chartArea;
  const abs = (x, y) => ({ x: r.left + x, y: r.top + y });
  const xAxisY = Math.min(r.height - 12, a.bottom + Math.max(10, (r.height - a.bottom) * 0.5));
  const yAxisX = Math.max(12, a.left * 0.5);
  return {
    xAxisStart: abs(a.left + (a.right - a.left) * 0.22, xAxisY),
    xAxisEnd: abs(a.left + (a.right - a.left) * 0.78, xAxisY),
    yAxisStart: abs(yAxisX, a.bottom - (a.bottom - a.top) * 0.22),
    yAxisEnd: abs(yAxisX, a.top + (a.bottom - a.top) * 0.78),
    rectStart: abs(a.left + (a.right - a.left) * 0.22, a.top + (a.bottom - a.top) * 0.22),
    rectEnd: abs(a.left + (a.right - a.left) * 0.78, a.top + (a.bottom - a.top) * 0.78),
    center: abs((a.left + a.right) / 2, (a.top + a.bottom) / 2),
    xCursor: abs((a.left + a.right) / 2, xAxisY),
    yCursor: abs(yAxisX, (a.top + a.bottom) / 2),
  };
}

async function cursorAt(page, point) {
  await page.mouse.move(point.x, point.y);
  await page.waitForTimeout(100);
  return page.evaluate((pt) => {
    const el = document.elementFromPoint(pt.x, pt.y);
    return {
      cursor: el ? getComputedStyle(el).cursor : null,
      tag: el?.tagName || null,
      id: el?.id || null,
      className: typeof el?.className === "string" ? el.className : "",
    };
  }, point);
}

function cursorFeedbackFailure(label, observed, allowed) {
  const cursor = observed.cursor || "";
  const expected = allowed.split(",").map((item) => item.trim()).filter(Boolean);
  const genericBad = new Set(["", "auto", "default", "initial", "inherit"]);
  if (genericBad.has(cursor)) return { message: `${label} cursor has no feedback`, details: observed };
  if (expected.length) {
    if (!expected.includes(cursor)) {
      return { message: `${label} cursor did not match expected values`, details: { observed, expected } };
    }
  }
  return null;
}

async function clickCrt(page, desiredOn) {
  const button = page.locator("#crt-toggle");
  if (!(await button.count())) return false;
  const pressed = await button.getAttribute("aria-pressed");
  const isOn = pressed === "true";
  if (isOn !== desiredOn) {
    await button.click();
    await page.waitForTimeout(500);
  }
  return true;
}

async function main() {
  fs.mkdirSync(ROOT, { recursive: true });
  const { chromium } = requirePlaywright();
  const browser = await chromium.launch({ headless: HEADLESS });
  const context = await browser.newContext({ viewport: VIEWPORT, deviceScaleFactor: 1 });
  const page = await context.newPage();
  const consoleErrors = [];
  const failures = [];
  const check = (condition, message, details = undefined) => {
    if (!condition) failures.push({ message, details });
  };
  const checkCursor = (label, observed, allowed) => {
    const failure = cursorFeedbackFailure(label, observed, allowed);
    if (failure) failures.push(failure);
  };

  page.on("console", (msg) => {
    if (msg.type() === "error") consoleErrors.push(`console.error: ${msg.text()}`);
  });
  page.on("pageerror", (err) => {
    consoleErrors.push(`pageerror: ${err.message}`);
  });

  await page.goto(URL, { waitUntil: "domcontentloaded", timeout: 60000 });
  await page.waitForLoadState("networkidle", { timeout: 60000 }).catch(() => {});
  await page.waitForFunction((sel) => {
    const canvas = document.querySelector(sel);
    return Boolean(canvas && window.Chart?.getChart?.(canvas));
  }, CHART_SELECTOR, { timeout: 60000 });

  await page.locator(CHART_SELECTOR).scrollIntoViewIfNeeded();
  await resetZoom(page);
  let state = await chartState(page);
  assert(state?.x && state?.y, "Selected chart is missing x/y scales", state);
  const pts = pointsFor(state);

  const xCursor = await cursorAt(page, pts.xCursor);
  const yCursor = await cursorAt(page, pts.yCursor);
  const centerCursor = await cursorAt(page, pts.center);
  checkCursor("x-axis", xCursor, process.env.EXPECT_CURSOR_X || "");
  checkCursor("y-axis", yCursor, process.env.EXPECT_CURSOR_Y || "");
  checkCursor("chart-center", centerCursor, process.env.EXPECT_CURSOR_CENTER || "");

  await resetZoom(page);
  const beforeXDrag = await chartState(page);
  await drag(page, pts.xAxisStart, pts.xAxisEnd);
  const afterXDrag = await chartState(page);
  check(scaleChanged(beforeXDrag, afterXDrag, "x"), "x-axis drag did not change x scale", { beforeXDrag, afterXDrag });
  check(scaleUnchanged(beforeXDrag, afterXDrag, "y"), "x-axis drag changed y scale", { beforeXDrag, afterXDrag });

  await resetZoom(page);
  state = await chartState(page);
  const pts2 = pointsFor(state);
  const beforeYDrag = await chartState(page);
  await drag(page, pts2.yAxisStart, pts2.yAxisEnd);
  const afterYDrag = await chartState(page);
  check(scaleChanged(beforeYDrag, afterYDrag, "y"), "y-axis drag did not change y scale", { beforeYDrag, afterYDrag });
  check(scaleUnchanged(beforeYDrag, afterYDrag, "x"), "y-axis drag changed x scale", { beforeYDrag, afterYDrag });

  await resetZoom(page);
  state = await chartState(page);
  const pts3 = pointsFor(state);
  const beforeRectDrag = await chartState(page);
  await drag(page, pts3.rectStart, pts3.rectEnd);
  const afterRectDrag = await chartState(page);
  check(scaleChanged(beforeRectDrag, afterRectDrag, "x"), "rect drag did not change x scale", { beforeRectDrag, afterRectDrag });
  check(scaleChanged(beforeRectDrag, afterRectDrag, "y"), "rect drag did not change y scale", { beforeRectDrag, afterRectDrag });

  await page.screenshot({ path: path.join(ROOT, SCREENSHOTS.interactions), fullPage: true });

  await resetZoom(page);
  state = await chartState(page);
  const pts4 = pointsFor(state);
  const beforeWheel = await chartState(page);
  await page.mouse.move(pts4.center.x, pts4.center.y);
  await page.mouse.wheel(0, -700);
  await page.waitForTimeout(250);
  const afterWheel = await chartState(page);
  check(
    scaleChanged(beforeWheel, afterWheel, "x") || scaleChanged(beforeWheel, afterWheel, "y"),
    "wheel did not change chart scale",
    { beforeWheel, afterWheel }
  );

  await resetZoom(page);
  await clickCrt(page, false);
  await page.screenshot({ path: path.join(ROOT, SCREENSHOTS.crtOff), fullPage: true });
  const crtToggleAvailable = await clickCrt(page, true);
  if (crtToggleAvailable) {
    await page.screenshot({ path: path.join(ROOT, SCREENSHOTS.crtOn), fullPage: true });
  }

  const fullscreenButton = page.locator(`[data-fullscreen][data-chart-id="${state.canvasId}"]`).first();
  let ratio = null;
  if (await fullscreenButton.count()) {
    await fullscreenButton.click();
    await page.waitForTimeout(700);
    const fullscreen = await chartState(page);
    ratio = (fullscreen.rect.width * fullscreen.rect.height) / (VIEWPORT.width * VIEWPORT.height);
    await page.screenshot({ path: path.join(ROOT, SCREENSHOTS.fullscreen), fullPage: false });
    check(ratio > 0.8, "Fullscreen canvas area is not >80% of the viewport", { ratio, fullscreen });
  } else {
    failures.push({ message: "Fullscreen button not found for selected chart", details: state });
  }

  check(consoleErrors.length === 0, "Console/page errors were observed", consoleErrors);

  console.log(JSON.stringify({
    ok: failures.length === 0,
    url: URL,
    chartSelector: CHART_SELECTOR,
    chartCanvasId: state.canvasId,
    screenshots: Object.fromEntries(Object.entries(SCREENSHOTS).map(([key, name]) => [key, path.join(ROOT, name)])),
    cursors: { xAxis: xCursor.cursor, yAxis: yCursor.cursor, center: centerCursor.cursor },
    fullscreenCanvasViewportRatio: ratio === null ? null : Number(ratio.toFixed(4)),
    failures,
  }, null, 2));

  await browser.close();
  if (failures.length) process.exit(1);
}

main().catch((err) => {
  console.error(err.stack || err.message || err);
  process.exit(1);
});
