# Codex review: v6 documentation consistency audit — 2026-05-06

## Finding 1

- Severity: HIGH
- File: `README.md:19`, `README.md:29`, `docs/superpowers/experiments/2026-05-05-v6-architecture-canonical.md:45`, `docs/superpowers/experiments/2026-05-05-v6-architecture-canonical.md:239`
- Description: The docs describe v6 as if the persistent canvas is warped, cross-attended, rasterized, updated, and reused for OSS-FX. The current `oss/sr/v6/model.py` forward path does not do that: `motion_lr`, depth inputs, and `frame_index` are unused; `_canvas_state` is never populated; an empty canvas returns zero tokens; fusion short-circuits to identity; output is produced by a pixel-shuffle head. `tests/sr/v6/test_model.py:5` also states the canvas-update path is not wired.
- Suggested fix: Add a "current implementation status" note wherever the architecture is summarized: current v6 code is HAT + empty-canvas identity fusion + pixel head; canvas write/warp/rasterizer/FX integration remains pending. Keep the full diagram, but label it target architecture.

## Finding 2

- Severity: HIGH
- File: `docs/superpowers/experiments/2026-05-05-v6-architecture-canonical.md:17`, `docs/superpowers/experiments/2026-05-05-v6-architecture-canonical.md:211`, `docs/superpowers/experiments/2026-05-05-v6-architecture-canonical.md:225`
- Description: These lines claim or imply OSS-SR ships through DLL-shim drop-in today, works on hundreds of AAA games today, or will be day-one usable by dropping a DLL into a game directory. That contradicts `README.md:79`, which says no game integration has shipped and the DXGI/NGX shim is designed but not built.
- Suggested fix: Reframe as planned S7 behavior only. Example: "The planned integration path is a DLL shim for titles already exposing DLSS/FSR/XeSS inputs. No game integration has shipped yet; the listed games are candidate validation targets."

## Finding 3

- Severity: MEDIUM
- File: `README.md:71`, `README.md:89`, `RESEARCH.md:19`, `RESEARCH.md:170`, `RESEARCH.md:178`
- Description: Status and result sections are stale relative to the included 2026-05-06 v5 held-out memo and current v6 code. README still says v5 quality numbers do not exist and pixel-temporal training has not finished. RESEARCH says v5 is still in training/queued and v6 implementation has not started / is queued. The reviewed set includes `docs/superpowers/experiments/2026-05-06-v5-pixel-temporal-final-held-out-eval.md:111`, which reports the final v5-pixel corrected held-out result, and `oss/sr/v6/` now contains landed v6 modules.
- Suggested fix: Update status to: v5-pixel-temporal completed with PSNR 25.703 / LPIPS 0.1666 / temporal ratio 0.337 on the TartanAir oldtown held-out batch; v5-Gaussian-temporal is no longer the main baseline path unless staged smoke tests continue; v6 modules and orchestrator exist, but training loop and full canvas/rasterizer wiring are still incomplete.

## Finding 4

- Severity: MEDIUM
- File: `docs/superpowers/experiments/2026-05-05-v6-architecture-canonical.md:93`, `docs/superpowers/experiments/2026-05-05-v6-architecture-canonical.md:107`, `README.md:132`
- Description: The performance framing has arithmetic and support problems. Line 107 says GRAPE at 69.33 FPS "hits the <3 ms inference budget"; 69.33 FPS is about 14.4 ms per frame. Lines 93 and README 132 say Pico undercuts FSR 2/3, but the same docs list Pico as `<2 ms` and FSR 2/3 as roughly `0.4-1.0 ms`, so "undercuts" is not supported.
- Suggested fix: Replace with neutral target language. For GRAPE, say it demonstrates a compact Gaussian predictor in a relevant footprint, not that it hits the v6 latency budget. For Pico vs FSR, say the target is to be in the handheld budget band, with measurements pending.

## Finding 5

- Severity: MEDIUM
- File: `docs/superpowers/experiments/2026-05-05-v6-architecture-canonical.md:63`, `docs/superpowers/experiments/2026-05-05-v6-architecture-canonical.md:89`, `docs/research/2026-05-05-v6-external-baselines-integration-plan.md:63`, `docs/research/2026-05-06-v6-source-extraction-notes.md:167`
- Description: The HAT teacher naming and parameter count disagree. The canonical memo calls the teacher "HAT-Base" / "HAT-Base (~15M params)" and the plan says HAT-Base and HAT-L are different parameterizations. Source notes say HAT-L is ~40M params and should mirror the upstream YAML exactly. Current code aliases `"hat-base"` to `hat_l`, but `oss/sr/v6/hat.py:303` implements a trimmed HAT-L-derived model with `depth=6` and `blocks_per_group=5`, not the upstream HAT-L `[6]*12` config.
- Suggested fix: Pick one naming scheme. If code is authoritative, document Heavy as "OSS HAT-L-derived/trimmed Heavy (~17M target)" and stop calling it upstream HAT-L parity or HAT-Base. If upstream HAT-L warm-start is required, update code and tables to the real upstream HAT-L size.

## Finding 6

- Severity: MEDIUM
- File: `docs/research/2026-05-05-v6-external-baselines-integration-plan.md:18`, `docs/research/2026-05-06-v6-source-extraction-notes.md:39`, `docs/research/2026-05-06-v6-source-extraction-notes.md:47`, `docs/research/2026-05-06-v6-source-extraction-notes.md:544`
- Description: The integration plan states GSASR has a RoPE fusion module to adopt, while the source extraction notes say the paper describes Swin-style learnable relative position bias and the actual `fea2gsropeamp_arch.py` source could not be retrieved. Current OSS code uses RoPE in `oss/sr/v6/cross_attention.py`, so the docs blur "GSASR confirmed behavior" and "OSS adaptation."
- Suggested fix: Change the plan to "adopt GSASR-style window cross-attention; RoPE is unconfirmed upstream and currently an OSS adaptation pending source inspection." Keep the clone/inspect action as the gate before claiming upstream parity.

## Finding 7

- Severity: MEDIUM
- File: `docs/research/2026-05-06-v6-source-extraction-notes.md:424`, `docs/research/2026-05-06-v6-source-extraction-notes.md:455`, `docs/research/2026-05-06-v6-source-extraction-notes.md:484`, `docs/research/2026-05-06-v6-source-extraction-notes.md:558`
- Description: The DLSS-RR section presents the seven-slot G-buffer and motion-vector convention as the contract, but the same file says the method was WebFetch/WebSearch only and that the motion-vector sign convention still needs shader inspection. "The per-frame G-buffer required is exactly the seven slots above" is too strong until the sample is cloned and the writing shaders are checked.
- Suggested fix: Mark the seven slots as the currently inferred `vk_gaussian_splatting` sample binding layout, not a verified DLSS-RR contract. Keep motion-vector sign/scale/jitter handling explicitly unresolved until the sample shaders are inspected.

## Finding 8

- Severity: LOW
- File: `docs/superpowers/experiments/2026-05-05-v6-architecture-canonical.md:211`, `docs/superpowers/experiments/2026-05-05-v6-architecture-canonical.md:281`, `docs/superpowers/experiments/2026-05-05-v6-architecture-canonical.md:290`, `docs/research/2026-05-05-v6-external-baselines-integration-plan.md:23`, `RESEARCH.md:153`, `RESEARCH.md:197`, `docs/superpowers/experiments/2026-05-06-v5-pixel-temporal-final-held-out-eval.md:153`, `docs/superpowers/experiments/2026-05-06-v5-pixel-temporal-final-held-out-eval.md:159`
- Description: Forbidden rhetorical qualifiers from `docs/coordination/codex-project-context.md` appear in reviewed docs. The clear cases are `killer`, `honest` / `honestly`, and quoted `real` as a performative qualifier. Several `actual` occurrences also appear in reviewed files, including `docs/research/2026-05-05-v6-external-baselines-integration-plan.md:23` and `docs/research/2026-05-05-v6-external-baselines-integration-plan.md:204`.
- Suggested fix: Replace with neutral wording: "integration path", "risks", "likely", "measured/native performance", "current contribution", or simply delete the qualifier.

Summary: HIGH=2 MEDIUM=5 LOW=1
