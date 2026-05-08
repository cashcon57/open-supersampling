#!/usr/bin/env node
"use strict";

const fs = require("node:fs");
const http = require("node:http");
const path = require("node:path");
const { execSync } = require("node:child_process");

const ROOT = process.env.OSS_ROOT || "/Users/cashconway/OpenSuperSampling";
const DASHBOARD_DIR = path.join(ROOT, "dashboard-public");
const DATA_PATH = path.join(DASHBOARD_DIR, "data.json");
const HEADLESS = process.env.HEADLESS !== "0";

function requirePlaywright() {
  const candidates = [];
  if (process.env.PLAYWRIGHT_MODULE) candidates.push(process.env.PLAYWRIGHT_MODULE);
  try {
    candidates.push(require.resolve("playwright"));
  } catch {}
  candidates.push(path.join(ROOT, "tests/playwright-3080ti/node_modules/playwright"));
  try {
    candidates.push(path.join(execSync("npm root -g", { encoding: "utf8" }).trim(), "playwright"));
  } catch {}
  for (const candidate of candidates) {
    try {
      return require(candidate);
    } catch {}
  }
  throw new Error("Could not load Playwright. Install it globally or set PLAYWRIGHT_MODULE.");
}

function assert(condition, message, details = undefined) {
  if (!condition) {
    const suffix = details === undefined ? "" : `\n${JSON.stringify(details, null, 2)}`;
    throw new Error(`${message}${suffix}`);
  }
}

function contentType(filePath) {
  if (filePath.endsWith(".html")) return "text/html; charset=utf-8";
  if (filePath.endsWith(".js")) return "application/javascript; charset=utf-8";
  if (filePath.endsWith(".json")) return "application/json; charset=utf-8";
  if (filePath.endsWith(".svg")) return "image/svg+xml";
  if (filePath.endsWith(".png")) return "image/png";
  if (filePath.endsWith(".css")) return "text/css; charset=utf-8";
  return "application/octet-stream";
}

function serveDashboard() {
  const server = http.createServer((request, response) => {
    const url = new URL(request.url || "/", "http://127.0.0.1");
    const pathname = decodeURIComponent(url.pathname === "/" ? "/index.html" : url.pathname);
    const candidate = path.normalize(path.join(DASHBOARD_DIR, pathname));
    if (!candidate.startsWith(DASHBOARD_DIR)) {
      response.writeHead(403);
      response.end("forbidden");
      return;
    }
    fs.readFile(candidate, (error, body) => {
      if (error) {
        response.writeHead(404);
        response.end("not found");
        return;
      }
      response.writeHead(200, { "content-type": contentType(candidate), "cache-control": "no-store" });
      response.end(body);
    });
  });
  return new Promise((resolve) => {
    server.listen(0, "127.0.0.1", () => {
      const address = server.address();
      resolve({ server, url: `http://127.0.0.1:${address.port}/` });
    });
  });
}

function withSyntheticLoss(diverged) {
  const data = JSON.parse(fs.readFileSync(DATA_PATH, "utf8"));
  data.generated_at = new Date().toISOString();
  const run = (data.runs || []).find((item) => item.active) || data.runs?.[0];
  assert(run, "Fixture data.json does not contain a run");
  assert(Array.isArray(run.loss_curve) && run.loss_curve.length >= 100, "Fixture run needs at least 100 loss rows");

  run.loss_curve = run.loss_curve.map((row) => ({ ...row }));
  const start = run.loss_curve.length - 100;
  for (let index = start; index < run.loss_curve.length; index += 1) {
    const row = run.loss_curve[index];
    const baseline = index % 2 === 0 ? 0.95 : 1.05;
    row.loss_total = index === run.loss_curve.length - 1
      ? (diverged ? 1.3 : 1.02)
      : baseline;
  }

  const latest = run.loss_curve[run.loss_curve.length - 1];
  run.latest_step = Number(latest.step || run.latest_step || 0);
  run.latest_metrics = { ...(run.latest_metrics || {}), ...latest };
  return data;
}

async function main() {
  const { chromium } = requirePlaywright();
  const { server, url } = await serveDashboard();
  const browser = await chromium.launch({ headless: HEADLESS });
  const consoleErrors = [];
  let diverged = false;

  try {
    const context = await browser.newContext({ viewport: { width: 1440, height: 900 }, deviceScaleFactor: 1 });
    await context.route("**/data.json*", (route) => {
      route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify(withSyntheticLoss(diverged)),
      });
    });
    const page = await context.newPage();
    page.on("console", (msg) => {
      if (msg.type() === "error") consoleErrors.push(`console.error: ${msg.text()}`);
    });
    page.on("pageerror", (error) => consoleErrors.push(`pageerror: ${error.message}`));

    await page.goto(url, { waitUntil: "domcontentloaded", timeout: 60000 });
    await page.waitForFunction(() => document.querySelector("#data-state")?.textContent === "live", { timeout: 60000 });
    assert(!/DIVERGED/.test(await page.title()), "Baseline fixture should not start diverged", { title: await page.title() });

    diverged = true;
    await page.evaluate(async () => {
      await window.pollDashboardIdle();
    });
    await page.waitForFunction(() => /DIVERGED/.test(document.title), { timeout: 5000 });

    const bannerText = await page.locator("#loss-divergence-banner").textContent();
    assert(/Loss diverged at step/.test(bannerText || ""), "Divergence banner did not render", { bannerText });
    assert(!consoleErrors.length, "Console errors were reported", consoleErrors);

    await context.close();
    console.log("loss-divergence alarm verifier passed");
  } finally {
    await browser.close().catch(() => {});
    server.close();
  }
}

main().catch((error) => {
  console.error(error.stack || error.message || String(error));
  process.exit(1);
});
