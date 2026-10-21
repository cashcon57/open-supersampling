import { expect, test } from "@playwright/test";
import path from "node:path";

test("local Chrome loads the dashboard and visual progress", async ({ page }, testInfo) => {
  await page.goto("/", { waitUntil: "domcontentloaded" });
  await expect(page).toHaveTitle(/OpenSuperSampling/);
  await expect(page.locator("#global-viz-section")).toBeVisible();

  const vizFiles = await page.evaluate(async () => {
    const response = await fetch("data.json", { cache: "no-store" });
    const data = await response.json();
    const run = data.runs.find((item: { name?: string }) => item.name === "srcnn-v7.0-pico-005");
    return run?.viz_pngs?.slice(-4) ?? [];
  });
  expect(vizFiles).toContain("step-00019000.png");

  await page.screenshot({
    path: path.join(testInfo.outputDir, "local-dashboard.png"),
    fullPage: false,
  });
});
