#!/usr/bin/env node
"use strict";

const fs = require("node:fs");
const path = require("node:path");
const { execSync } = require("node:child_process");

const ROOT = process.env.OSS_ROOT || process.cwd();
const BASE_URL = process.env.OSS_URL || "http://127.0.0.1:4173";
const HEADLESS = process.env.HEADLESS !== "0";
const ARTIFACT_DIR = process.env.OSS_ARTIFACT_DIR || ROOT;
const DESKTOP = { width: 1440, height: 900 };
const MOBILE = { width: 390, height: 844 };
const SCREENSHOTS = {
  defaultDesktop: "oss-f2-url-state-default-desktop.png",
  deeplinkedDesktop: "oss-f2-url-state-deeplinked-desktop.png",
  mobile: "oss-f2-url-state-mobile.png",
};
const DEFAULT_PARAMS = {
  run: process.env.OSS_F2_RUN || "v6.1",
  step: process.env.OSS_F2_STEP || "5000",
  chart: process.env.OSS_F2_CHART || "loss-decomp",
  zoom: process.env.OSS_F2_ZOOM || "1500-3000",
};
const ALT_PARAMS = {
  run: process.env.OSS_F2_ALT_RUN || DEFAULT_PARAMS.run,
  step: process.env.OSS_F2_ALT_STEP || "7000",
  chart: process.env.OSS_F2_ALT_CHART || "loss",
  zoom: process.env.OSS_F2_ALT_ZOOM || "2000-4000",
};
const COPY_SELECTORS = [
  process.env.OSS_F2_COPY_SELECTOR,
  "[data-copy-url-state]",
  "[data-copy-deeplink]",
  "[data-share-url]",
  "[data-share-link]",
  "[data-chart-id='loss-curve'] [data-copy-link]",
  "[data-copy-link]",
  "button[aria-label*='Copy']",
  "button[aria-label*='copy']",
  "button[title*='Copy']",
  "button[title*='copy']",
].filter(Boolean);

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

function fail(message, details = undefined) {
  const suffix = details === undefined ? "" : `\n${JSON.stringify(details, null, 2)}`;
  throw new Error(`${message}${suffix}`);
}

function makeUrl(params) {
  const url = new URL(BASE_URL);
  for (const [key, value] of Object.entries(params)) {
    url.searchParams.set(key, value);
  }
  return url.toString();
}

function pathSearchFromParams(params) {
  const url = new URL(makeUrl(params));
  return `${url.pathname}${url.search}${url.hash}`;
}

function parseZoom(value) {
  if (Array.isArray(value) && value.length >= 2) return value.map(Number).slice(0, 2);
  if (value && typeof value === "object") {
    const min = value.min ?? value.start ?? value.from ?? value[0];
    const max = value.max ?? value.end ?? value.to ?? value[1];
    return [Number(min), Number(max)];
  }
  const match = String(value ?? "").match(/(-?\d+(?:\.\d+)?)\D+(-?\d+(?:\.\d+)?)/);
  return match ? [Number(match[1]), Number(match[2])] : [NaN, NaN];
}

function approxEqual(actual, expected, tolerance = 1e-6) {
  return Number.isFinite(actual) && Number.isFinite(expected) && Math.abs(actual - expected) <= tolerance;
}

function stringMatches(actual, expected) {
  if (actual === undefined || actual === null) return false;
  const text = typeof actual === "object"
    ? [actual.id, actual.name, actual.slug, actual.key, actual.label, actual.value].filter(Boolean).join(" ")
    : String(actual);
  return text === expected || text.includes(expected);
}

function extractCandidate(state, keys) {
  if (!state || typeof state !== "object") return undefined;
  for (const key of keys) {
    if (state[key] !== undefined) return state[key];
  }
  for (const key of ["url", "query", "params", "selection", "dashboard", "state"]) {
    if (state[key] && typeof state[key] === "object") {
      const found = extractCandidate(state[key], keys);
      if (found !== undefined) return found;
    }
  }
  return undefined;
}

function stateMatchesParams(state, params) {
  const run = extractCandidate(state, ["run", "runId", "runName", "selectedRun", "activeRun"]);
  const step = extractCandidate(state, ["step", "selectedStep", "currentStep"]);
  const chart = extractCandidate(state, ["chart", "chartId", "selectedChart", "activeChart"]);
  const zoom = extractCandidate(state, ["zoom", "xZoom", "xRange", "chartZoom"]);
  const [zoomMin, zoomMax] = parseZoom(zoom);
  const [expectedMin, expectedMax] = parseZoom(params.zoom);

  return {
    ok: stringMatches(run, params.run) &&
      Number(step) === Number(params.step) &&
      stringMatches(chart, params.chart) &&
      approxEqual(zoomMin, expectedMin) &&
      approxEqual(zoomMax, expectedMax),
    observed: { run, step, chart, zoom },
    expected: params,
  };
}

async function dashboardState(page) {
  return page.evaluate(() => {
    if (typeof window.__getDashboardState !== "function") return null;
    return window.__getDashboardState();
  });
}

async function waitForDashboard(page) {
  await page.waitForLoadState("domcontentloaded", { timeout: 60000 });
  await page.waitForFunction(() => {
    const state = document.getElementById("data-state")?.textContent || "";
    const hasRuns = Boolean(document.querySelector("#run-history details[data-run-name], #run-history details"));
    return /live|retrying/.test(state) || hasRuns;
  }, { timeout: 60000 });
}

async function assertState(page, params, label) {
  await page.waitForFunction(() => typeof window.__getDashboardState === "function", { timeout: 10000 })
    .catch(() => fail("window.__getDashboardState() is not defined", { label }));
  await page.waitForFunction((expected) => {
    const state = window.__getDashboardState();
    const pick = (obj, keys) => {
      for (const key of keys) if (obj?.[key] !== undefined) return obj[key];
      for (const key of ["url", "query", "params", "selection", "dashboard", "state"]) {
        if (obj?.[key] && typeof obj[key] === "object") {
          const found = pick(obj[key], keys);
          if (found !== undefined) return found;
        }
      }
      return undefined;
    };
    const textMatch = (actual, target) => {
      if (actual === undefined || actual === null) return false;
      const text = typeof actual === "object"
        ? [actual.id, actual.name, actual.slug, actual.key, actual.label, actual.value].filter(Boolean).join(" ")
        : String(actual);
      return text === target || text.includes(target);
    };
    const zoomParts = (value) => {
      if (Array.isArray(value) && value.length >= 2) return value.map(Number).slice(0, 2);
      if (value && typeof value === "object") return [Number(value.min ?? value.start ?? value.from ?? value[0]), Number(value.max ?? value.end ?? value.to ?? value[1])];
      const match = String(value ?? "").match(/(-?\d+(?:\.\d+)?)\D+(-?\d+(?:\.\d+)?)/);
      return match ? [Number(match[1]), Number(match[2])] : [NaN, NaN];
    };
    const [z0, z1] = zoomParts(pick(state, ["zoom", "xZoom", "xRange", "chartZoom"]));
    const [e0, e1] = zoomParts(expected.zoom);
    return textMatch(pick(state, ["run", "runId", "runName", "selectedRun", "activeRun"]), expected.run) &&
      Number(pick(state, ["step", "selectedStep", "currentStep"])) === Number(expected.step) &&
      textMatch(pick(state, ["chart", "chartId", "selectedChart", "activeChart"]), expected.chart) &&
      Math.abs(z0 - e0) <= 1e-6 &&
      Math.abs(z1 - e1) <= 1e-6;
  }, params, { timeout: 10000 }).catch(async () => {
    const state = await dashboardState(page);
    const match = stateMatchesParams(state, params);
    fail(`Dashboard state did not match ${label}`, match);
  });
}

async function assertDefaultPath(page) {
  await page.waitForFunction(() => typeof window.__getDashboardState === "function", { timeout: 10000 })
    .catch(() => fail("window.__getDashboardState() is not defined on the default path"));
  const state = await dashboardState(page);
  if (!state || typeof state !== "object") {
    fail("Default path did not expose an object dashboard state", { state });
  }
  const url = new URL(page.url());
  if (url.search) {
    fail("Default path unexpectedly loaded with query parameters", { url: page.url() });
  }
  return state;
}

async function chartScale(page, params) {
  return page.evaluate((expected) => {
    const selectors = {
      "loss-decomp": "[id^='loss-decomp-chart']",
      loss: "[id^='loss-chart']",
      "psnr-live": "[id^='psnr-chart']",
      "lpips-live": "[id^='lpips-chart']",
      "psnr-held-out": "[id^='heldout-chart']",
      "lpips-held-out": "[id^='heldout-chart']",
      "cross-version-psnr": "#cross-version-psnr-chart",
      "cross-version-lpips": "#cross-version-lpips-chart",
    };
    const selector = selectors[expected.chart] || `[id*='${expected.chart}']`;
    const canvas = document.querySelector(selector);
    const chart = canvas ? window.Chart?.getChart?.(canvas) : null;
    const scale = chart?.scales?.x;
    return {
      selector,
      found: Boolean(canvas),
      chart: canvas?.id || null,
      x: scale ? { min: Number(scale.min), max: Number(scale.max) } : null,
    };
  }, params);
}

async function assertZoomApplied(page, params, label) {
  const [expectedMin, expectedMax] = parseZoom(params.zoom);
  const scale = await chartScale(page, params);
  if (!scale.found || !scale.x) {
    fail(`Could not inspect chart x-scale for ${label}`, scale);
  }
  const tolerance = Math.max(1, Math.abs(expectedMax - expectedMin) * 0.01);
  if (!approxEqual(Number(scale.x.min), expectedMin, tolerance) || !approxEqual(Number(scale.x.max), expectedMax, tolerance)) {
    fail(`Chart zoom was not applied for ${label}`, { scale, expected: { min: expectedMin, max: expectedMax } });
  }
}

async function assertStepReferenceDrawn(page, params, label) {
  const result = await page.evaluate((expected) => {
    const card = document.querySelector(`[data-chart-slug="${expected.chart}"], [data-chart-id="${expected.chart}"]`);
    const canvas = card?.querySelector("canvas");
    const chart = canvas ? window.Chart?.getChart?.(canvas) : null;
    return {
      cardFound: Boolean(card),
      visible: Boolean(card && card.getBoundingClientRect().width && card.getBoundingClientRect().height),
      canvasId: canvas?.id || null,
      step: chart?.$urlStateStep ?? null,
      lineDrawn: Boolean(chart?.$urlStateStepLineDrawn),
    };
  }, params);
  if (!result.cardFound || !result.visible || Number(result.step) !== Number(params.step) || !result.lineDrawn) {
    fail(`Step reference line was not drawn for ${label}`, result);
  }
}

async function clickCopyButton(page) {
  for (const selector of COPY_SELECTORS) {
    const button = page.locator(selector).first();
    if (await button.count()) {
      await button.click();
      return selector;
    }
  }
  fail("No URL-state clipboard copy button found", { tried: COPY_SELECTORS });
}

async function assertClipboardCopy(page, params) {
  await page.evaluate(() => navigator.clipboard.writeText(""));
  const selector = await clickCopyButton(page);
  await page.waitForFunction(() => navigator.clipboard.readText().then(Boolean), { timeout: 5000 });
  const text = await page.evaluate(() => navigator.clipboard.readText());
  const copied = new URL(text, BASE_URL);
  for (const [key, value] of Object.entries(params)) {
    if (copied.searchParams.get(key) !== String(value)) {
      fail("Clipboard URL did not preserve the active F2 params", { selector, text, expected: params });
    }
  }
  return { selector, text };
}

async function runDesktop(browser, consoleErrors) {
  const context = await browser.newContext({
    viewport: DESKTOP,
    deviceScaleFactor: 1,
    permissions: ["clipboard-read", "clipboard-write"],
  });
  await context.grantPermissions(["clipboard-read", "clipboard-write"], { origin: new URL(BASE_URL).origin }).catch(() => {});
  const page = await context.newPage();
  page.on("console", (msg) => {
    if (msg.type() === "error") consoleErrors.push(`console.error: ${msg.text()}`);
  });
  page.on("pageerror", (err) => consoleErrors.push(`pageerror: ${err.message}`));

  await page.goto(BASE_URL, { waitUntil: "domcontentloaded", timeout: 60000 });
  await waitForDashboard(page);
  const defaultState = await assertDefaultPath(page);
  await page.screenshot({ path: path.join(ARTIFACT_DIR, SCREENSHOTS.defaultDesktop), fullPage: true });

  await page.goto(makeUrl(DEFAULT_PARAMS), { waitUntil: "domcontentloaded", timeout: 60000 });
  await waitForDashboard(page);
  await assertState(page, DEFAULT_PARAMS, "deeplink");
  await assertZoomApplied(page, DEFAULT_PARAMS, "deeplink");
  await assertStepReferenceDrawn(page, DEFAULT_PARAMS, "deeplink");
  await page.screenshot({ path: path.join(ARTIFACT_DIR, SCREENSHOTS.deeplinkedDesktop), fullPage: true });

  await page.goto(BASE_URL, { waitUntil: "domcontentloaded", timeout: 60000 });
  await waitForDashboard(page);
  await assertDefaultPath(page);
  const copyButton = page.locator("[data-chart-id='loss-curve'] [data-copy-link]").first();
  if (!(await copyButton.count())) fail("Required loss-curve copy-link button not found");
  await page.evaluate(() => navigator.clipboard.writeText(""));
  await copyButton.click();
  await page.waitForFunction(() => location.search.includes("chart=loss"), { timeout: 5000 });
  const url1 = page.url();
  if (!new URL(url1).searchParams.get("run")) fail("Copy-link URL did not include run", { url1 });
  await page.goBack({ waitUntil: "domcontentloaded", timeout: 15000 }).catch(() => {});
  if (new URL(page.url()).search) fail("Back navigation did not restore the bare URL", { url: page.url() });
  await page.goForward({ waitUntil: "domcontentloaded", timeout: 15000 }).catch(() => {});
  if (page.url() !== url1) fail("Forward navigation did not restore the copied URL", { expected: url1, actual: page.url() });
  const clipboardText = await page.evaluate(() => navigator.clipboard.readText());
  if (clipboardText !== page.url()) fail("Clipboard text did not match the restored copied URL", { clipboardText, url: page.url() });
  const clipboard = { selector: "[data-chart-id='loss-curve'] [data-copy-link]", text: clipboardText };

  await context.close();
  return { defaultState, clipboard };
}

async function runMobile(browser, consoleErrors) {
  const context = await browser.newContext({ viewport: MOBILE, deviceScaleFactor: 1, isMobile: true });
  const page = await context.newPage();
  page.on("console", (msg) => {
    if (msg.type() === "error") consoleErrors.push(`console.error: ${msg.text()}`);
  });
  page.on("pageerror", (err) => consoleErrors.push(`pageerror: ${err.message}`));
  await page.goto(makeUrl(DEFAULT_PARAMS), { waitUntil: "domcontentloaded", timeout: 60000 });
  await waitForDashboard(page);
  await assertState(page, DEFAULT_PARAMS, "mobile deeplink");
  await page.screenshot({ path: path.join(ARTIFACT_DIR, SCREENSHOTS.mobile), fullPage: true });
  await context.close();
}

async function main() {
  fs.mkdirSync(ARTIFACT_DIR, { recursive: true });
  const { chromium } = requirePlaywright();
  const browser = await chromium.launch({ headless: HEADLESS });
  const consoleErrors = [];
  let desktopResult;

  try {
    desktopResult = await runDesktop(browser, consoleErrors);
    await runMobile(browser, consoleErrors);
  } finally {
    await browser.close();
  }

  if (consoleErrors.length) {
    fail("Console/page errors were observed", consoleErrors);
  }

  console.log(JSON.stringify({
    ok: true,
    url: BASE_URL,
    defaultParams: DEFAULT_PARAMS,
    alternateParams: ALT_PARAMS,
    screenshots: {
      defaultDesktop: path.join(ARTIFACT_DIR, SCREENSHOTS.defaultDesktop),
      deeplinkedDesktop: path.join(ARTIFACT_DIR, SCREENSHOTS.deeplinkedDesktop),
      mobile: path.join(ARTIFACT_DIR, SCREENSHOTS.mobile),
    },
    clipboard: desktopResult.clipboard,
    defaultState: desktopResult.defaultState,
  }, null, 2));
}

main().catch((err) => {
  console.error(err.stack || err.message || err);
  process.exit(1);
});
