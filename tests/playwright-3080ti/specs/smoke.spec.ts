import { chromium, expect, test } from "@playwright/test";
import path from "node:path";

test("dashboard loads through 3080 Ti Chromium over CDP", async ({}, testInfo) => {
  const cdpURL =
    String(testInfo.project.metadata.cdpURL ?? "") ||
    process.env.OSS_3080TI_CDP_URL ||
    "http://3080ti-windows:9222";
  const baseURL =
    String(testInfo.project.use.baseURL ?? "") ||
    process.env.OSS_3080TI_BASE_URL ||
    "/";

  const browser = await chromium.connectOverCDP(cdpURL);
  const context = await browser.newContext({
    baseURL,
    viewport: { width: 1440, height: 900 },
  });

  try {
    const page = await context.newPage();
    await page.goto("/", { waitUntil: "domcontentloaded" });
    await expect(page).toHaveTitle(/OpenSuperSampling/);
    await page.screenshot({
      fullPage: true,
      path: path.join(process.cwd(), "test-results", "smoke-3080ti.png"),
    });
  } finally {
    await context.close();
  }
});
