#!/usr/bin/env node
"use strict";

const path = require("node:path");
const { execSync } = require("node:child_process");

const ROOT = process.env.OSS_ROOT || "/Users/cashconway/OpenSuperSampling";
const URL = process.env.OSS_URL || "https://opensupersampling.org";
const HEADLESS = process.env.HEADLESS !== "0";
const RUN_SELECTOR = 'details[data-run-name="srcnn-v6.1-pico-001"]';
const SCREENSHOTS = {
  twoCol: "oss-viz-blowup-2col-desktop.png",
  threeCol: "oss-viz-blowup-3col-desktop.png",
  mobile: "oss-viz-blowup-mobile-stack.png",
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
  throw new Error("Could not load Playwright. Install it globally or set PLAYWRIGHT_MODULE.");
}

function assert(condition, message, details = undefined) {
  if (!condition) {
    const suffix = details === undefined ? "" : `\n${JSON.stringify(details, null, 2)}`;
    throw new Error(`${message}${suffix}`);
  }
}

function matrixScale(transform) {
  const match = /matrix\(([^,]+)/.exec(transform || "");
  return match ? Number(match[1]) : 1;
}

async function installErrorCapture(page, errors) {
  page.on("console", (msg) => {
    if (msg.type() === "error") errors.push(msg.text());
  });
  page.on("pageerror", (error) => errors.push(error.message));
}

async function openRun(page) {
  await page.goto(URL, { waitUntil: "networkidle" });
  const run = page.locator(RUN_SELECTOR);
  await run.waitFor({ state: "attached", timeout: 15000 });
  const open = await run.evaluate((node) => node.open);
  if (!open) await run.locator("summary").click();
  await page.waitForSelector(`${RUN_SELECTOR} [data-viz-compare-chips] .viz-column-chip`);
  return run;
}

async function setSelectedColumns(run, indices) {
  const chips = run.locator("[data-viz-compare-chips] .viz-column-chip");
  const count = await chips.count();
  assert(count === 7, "Expected the v6.1 viz compare chip row to render 7 chips", { count });
  for (let i = 0; i < count; i += 1) {
    const chip = chips.nth(i);
    const checked = await chip.getAttribute("aria-checked");
    if (checked === "true" && !indices.includes(i)) await chip.click();
  }
  for (const index of indices) {
    const chip = chips.nth(index);
    const checked = await chip.getAttribute("aria-checked");
    if (checked !== "true") await chip.click();
  }
}

async function openCompare(run, indices) {
  await setSelectedColumns(run, indices);
  const cta = run.locator("[data-viz-compare-cta]");
  await cta.waitFor({ state: "visible" });
  assert(await cta.isEnabled(), "Compare CTA should be enabled with 2+ selected columns");
  await cta.click();
  await run.page().waitForSelector("#viz-compare-modal[open] .viz-compare-panel");
}

async function panelState(page) {
  return page.evaluate(() => Array.from(document.querySelectorAll(".viz-compare-panel-image")).map((node) => {
    const style = getComputedStyle(node);
    const rect = node.getBoundingClientRect();
    return {
      backgroundImage: style.backgroundImage,
      backgroundPosition: style.backgroundPosition,
      transform: style.transform,
      rect: { x: rect.x, y: rect.y, width: rect.width, height: rect.height },
    };
  }));
}

async function dragFirstPanel(page) {
  const box = await page.locator(".viz-compare-panel-stage").first().boundingBox();
  assert(box, "Could not locate first compare panel stage");
  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
  await page.mouse.down();
  await page.mouse.move(box.x + box.width / 2 + 80, box.y + box.height / 2 + 36, { steps: 8 });
  await page.mouse.up();
  await page.waitForTimeout(160);
}

async function verifyDesktop(browser, errors) {
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
  await installErrorCapture(page, errors);
  const run = await openRun(page);

  await openCompare(run, [0, 3]);
  const twoCol = await panelState(page);
  assert(twoCol.length === 2, "Expected 2 compare panels", twoCol);
  assert(twoCol[0].backgroundImage === twoCol[1].backgroundImage, "Compare panels should use the same viz PNG", twoCol);
  assert(twoCol[0].backgroundPosition !== twoCol[1].backgroundPosition, "Compare panel crops should use different background positions", twoCol);
  const panelRects = await page.evaluate(() => Array.from(document.querySelectorAll(".viz-compare-panel")).map((node) => node.getBoundingClientRect().toJSON()));
  assert(Math.abs(panelRects[0].y - panelRects[1].y) < 4, "Two selected columns should render side-by-side", panelRects);
  await page.screenshot({ path: path.join(ROOT, SCREENSHOTS.twoCol) });

  await dragFirstPanel(page);
  const afterDrag = await panelState(page);
  assert(afterDrag.every((panel) => panel.transform === afterDrag[0].transform), "Drag should synchronize transforms across panels", afterDrag);
  assert(afterDrag[0].transform !== twoCol[0].transform, "Drag should update the transform", { before: twoCol, after: afterDrag });

  const box = await page.locator(".viz-compare-panel-stage").first().boundingBox();
  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
  await page.mouse.wheel(0, -320);
  await page.waitForTimeout(160);
  const afterWheel = await panelState(page);
  const scales = afterWheel.map((panel) => matrixScale(panel.transform));
  assert(scales.every((scale) => Math.abs(scale - scales[0]) < 0.001), "Wheel zoom should synchronize scale across panels", afterWheel);
  assert(scales[0] > matrixScale(afterDrag[0].transform), "Wheel zoom should increase panel scale", { scales, afterDrag });

  await page.keyboard.press("Escape");
  await page.waitForFunction(() => !document.querySelector("#viz-compare-modal")?.open);

  await openCompare(run, [0, 1, 3]);
  const threeCol = await panelState(page);
  assert(threeCol.length === 3, "Expected 3 compare panels", threeCol);
  await page.screenshot({ path: path.join(ROOT, SCREENSHOTS.threeCol) });
  await page.keyboard.press("Escape");
  await page.close();
}

async function verifyMobile(browser, errors) {
  const page = await browser.newPage({
    viewport: { width: 390, height: 844 },
    isMobile: true,
    hasTouch: true,
  });
  await installErrorCapture(page, errors);
  const run = await openRun(page);
  await openCompare(run, [0, 1, 3]);
  const rects = await page.evaluate(() => Array.from(document.querySelectorAll(".viz-compare-panel")).map((node) => node.getBoundingClientRect().toJSON()));
  assert(rects.length === 3, "Expected 3 mobile compare panels", rects);
  assert(rects[1].y > rects[0].y && rects[2].y > rects[1].y, "Mobile compare panels should stack vertically", rects);
  await page.screenshot({ path: path.join(ROOT, SCREENSHOTS.mobile) });
  await page.keyboard.press("Escape");
  await page.close();
}

(async () => {
  const { chromium } = requirePlaywright();
  const browser = await chromium.launch({ headless: HEADLESS });
  const errors = [];
  try {
    await verifyDesktop(browser, errors);
    await verifyMobile(browser, errors);
    assert(errors.length === 0, "Console/page errors occurred during viz compare verification", errors);
  } finally {
    await browser.close();
  }
  console.log(JSON.stringify({ ok: true, url: URL, screenshots: SCREENSHOTS }, null, 2));
})().catch((error) => {
  console.error(error.stack || error.message || String(error));
  process.exit(1);
});
