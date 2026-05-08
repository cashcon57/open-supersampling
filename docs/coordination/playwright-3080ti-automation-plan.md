# Playwright 3080 Ti Automation Plan

**Goal:** Mac-side Playwright connecting via Chrome DevTools Protocol over Tailscale to a remote Chromium running on `3080ti-windows`. Real WebGPU + RTX 3080 Ti drivers do the rendering, so screenshots reflect what an actual GPU user sees.

## A. Approach

**Recommended: Option 2 — Mac-side Playwright, remote browser via CDP.**

3080 Ti runs Chromium with `--remote-debugging-port=9222 --remote-debugging-address=100.121.175.55 --user-data-dir=E:\playwright-profile`. Mac runs Playwright. Connect via `chromium.connectOverCDP("http://3080ti-windows:9222")`. Browser process + rendering on 3080 Ti; test logic on Mac.

Documented fallback to **Option 3 (hybrid)**: Playwright runner on 3080 Ti, dispatched from Mac over SSH. Use if CDP rejected by `Cross-Origin-Embedder-Policy` headers, auth flows, or download tests.

## B. Setup on the 3080 Ti

1. `npx playwright install chromium` with `PLAYWRIGHT_BROWSERS_PATH=E:\ms-playwright`.
2. Verify Node.js (already installed via wrangler).
3. New `scripts/3080ti/start-playwright-chromium.ps1`:
   - `--remote-debugging-port=9222`
   - `--remote-debugging-address=100.121.175.55` (CRITICAL: not `0.0.0.0`)
   - `--user-data-dir=E:\playwright-profile` (isolated)
   - `--enable-unsafe-webgpu --enable-features=Vulkan`
   - WMI orphan-spawn pattern (matches viz-daemon scripts).
4. New `scripts/3080ti/playwright-chromium-supervisor.ps1` polling every 60s.
5. Verify reachability: `curl http://3080ti-windows:9222/json/version`.

## C. Setup on the Mac

1. `tests/playwright-3080ti/package.json` with `"@playwright/test": "^1.49"`.
2. `tests/playwright-3080ti/playwright.config.ts` with `chromium.connectOverCDP(env.OSS_3080TI_CDP_URL ?? "http://3080ti-windows:9222")`.
3. First spec `specs/smoke.spec.ts`: connect → goto / → assert title → screenshot → close context (NOT `browser.close()`).

## D. Use case matrix

| Use case | Spec |
|---|---|
| Smoke: dashboard loads | `specs/smoke.spec.ts` |
| Chart axis drag-zoom on real GPU | `specs/axis-zoom.spec.ts` |
| B-viz-blowup checkbox compare flow | `specs/viz-blowup.spec.ts` |
| Full dashboard suite from 3080 Ti perspective | `specs/dashboard-suite.spec.ts` |
| WebGPU in-browser inference (when C1 ships) | `specs/webgpu-inference.spec.ts` |
| CRT shader (WebGL2 vs WebGPU) | `specs/crt-shader.spec.ts` |
| Periodic dashboard health probe | `specs/health-probe.spec.ts` |
| Fullscreen mode | `specs/fullscreen.spec.ts` |

## E. Phased rollout

| Phase | Scope | Days |
|---|---|---|
| P1 | MVP smoke setup (CDP reachable + first spec green) | 1 |
| P2 | Port existing visual baselines as snapshot specs | 2 |
| P3 | Hourly probe + status JSON publishing | 1 |
| P4 | WebGPU integration (post-C1) | TBD |
| P5 | CI integration (post-green hook on `ci_auto_heal`) | 0.5 |

## F. Boundaries / safety

1. **Bind to tailnet IP only** — `--remote-debugging-address=100.121.175.55`. CDP port is unauthenticated.
2. **Isolated user-data-dir** at `E:\playwright-profile` — never log into Google/GitHub/dashboard admin.
3. **No `file://` navigation** — spec helper rejects.
4. **No download trust** — sandbox dir + nightly sweep.
5. **Operator-presence gate** — probe `quser` idle-time before destructive interaction tests.
6. **No credential storage** in 3080 Ti profile.
7. **Kill-switch** `scripts/3080ti/stop-playwright-chromium.ps1`.

## G. Open questions for operator

1. **Interference with interactive use:** headless (default v1) vs headed in separate user session vs operator-asleep windows?
2. **Credential boundary:** confirm Playwright profile NEVER logs into Google/GitHub/admin?
3. **Scheduling window:** anytime vs operator-asleep (02:00-06:00)?
4. **Status-JSON publish target:** R2 bucket vs dashboard origin?
5. **Cleanup cadence:** screenshot retention 7 days?

## Critical files

- `scripts/3080ti/start-playwright-chromium.ps1` (NEW)
- `scripts/3080ti/playwright-chromium-supervisor.ps1` (NEW)
- `tests/playwright-3080ti/playwright.config.ts` (NEW)
- `tests/playwright-3080ti/specs/smoke.spec.ts` (NEW)
- `scripts/3080ti/launch-ci-heal-watch.ps1` (EXTEND with post-green Playwright hook in P5)
