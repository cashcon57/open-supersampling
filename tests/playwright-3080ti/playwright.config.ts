import { defineConfig } from "@playwright/test";
import { execFileSync } from "node:child_process";

const cdpURL = process.env.OSS_3080TI_CDP_URL ?? "http://3080ti-windows:9222";
const localPort = Number(process.env.OSS_3080TI_LOCAL_PORT ?? "39222");

function macTailnetIP(): string {
  try {
    return execFileSync("tailscale", ["ip", "-4"], { encoding: "utf8" }).trim().split(/\s+/)[0];
  } catch {
    return "127.0.0.1";
  }
}

const baseURL = process.env.OSS_3080TI_BASE_URL ?? `http://${macTailnetIP()}:${localPort}`;

export default defineConfig({
  testDir: "./specs",
  outputDir: "./test-results",
  reporter: [["html", { outputFolder: "playwright-report", open: "never" }], ["list"]],
  webServer: process.env.OSS_3080TI_BASE_URL
    ? undefined
    : {
        command: `python3 -m http.server ${localPort} --bind 0.0.0.0 --directory ../../dashboard-public`,
        url: `http://127.0.0.1:${localPort}`,
        reuseExistingServer: true,
        timeout: 10_000,
      },
  projects: [
    {
      name: "3080ti-chromium",
      metadata: { cdpURL },
      use: {
        baseURL,
      },
    },
  ],
});
