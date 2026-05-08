# Dashboard improvement backlog

Three-perspective brainstorm — operator (you), community (visitors), researcher (ML peers) + cross-cutting infra. 33 ideas + 5 user-added items (38 total). User-triaged 2026-05-07: each row marked YES / CONDITIONAL / DISCUSS. Use this as the backlog source when picking the next codex dispatch.

Brainstorm date: 2026-05-07. Cross-check against current state of `dashboard-public/index.html` before dispatching — some ideas may already have shipped.

## Legend

- ✅ YES — clear go, dispatch when capacity allows
- 🟡 CONDITIONAL — go-but-with-constraint (cost cap, public-safe content, etc.)
- 💬 DISCUSS — needs interactive Q&A before scoping. Don't dispatch yet.
- ❌ NO / DEFER — declined or paused
- 🆕 NEW — added by user during 2026-05-07 triage; not in original 33

## Operator perspective (Cash building OSS)

| # | Verdict | Idea | Why | Effort | Constraints / notes |
|---|---|---|---|---|---|
| O1 | ✅ | **Loss-divergence alarm.** Tab title flashes "⚠ DIVERGED" if loss_total grows >2σ over a 100-step window. Browser notification API for desktop. | You're not always watching; missing a NaN spiral costs hours. | Medium | — |
| O2 | 🟡 | **Per-step memo.** Click any point on the loss curve → write a markdown note attached to that step. Persists in repo at `docs/lab/<run>/step-<N>.md`. | Lab notebook discipline embedded in the dashboard. | Medium | Memos are PUBLIC. Any personal info excluded. Needs a "publish vs draft" gate before memos appear on the public site. |
| O3 | ✅ | **Live `train.log` tail panel.** Last 50 lines of the trainer's stdout, ANSI-colored, behind a collapsible drawer. | Currently you SSH or open the local dashboard for this. One source of truth. | Small | — |
| O4 | ✅ | **Diff view: ckpt A vs ckpt B.** Pick two saved ckpts, side-by-side viz strip + per-frame metric delta. | Detects subtle regression between adjacent ckpts. | Medium | Must be intuitive — visual UX test required. |
| O5 | 💬 | **Ckpt download buttons.** Click any ckpt marker → signed-URL download from R2. | Right now ckpts only live on 3080ti. Reproducibility for outsiders. | Medium (needs R2 ckpt sync) | Q for user: storage cost + how-to-publish-only-validated-ckpts. |
| O6 | 💬 | **Cost meter.** Cumulative training cost (kWh × electric rate + R2 storage + bandwidth). Live readout. | Solo-funding signal. | Small | Q for user: useful or noise? You said "you know your burn rate"; reconsider only if it pairs with R4 cost projection. |
| O7 | ✅ + 🆕 | **ETA to convergence + future-trend extrapolation line on active charts.** Linear/exponential fit on recent loss → target loss → ETA. **🆕 The trend line should EXTEND INTO THE FUTURE for active training, factoring in trajectory + decay rate.** | Replaces rough "approaching step-5K" mental math; gives visitors a forward-looking signal. | Small (ETA) + Medium (extrapolation) | Decay model: exponential or stretched-exp; render as dashed line beyond last data point with a confidence cone. |
| O8 | 🟡 | **One-click stop / pause / resume training** from the dashboard. | Faster reaction to anomalies. | Medium | **ONLY when visitor is on operator's tailnet** (IP gate via Worker). Hide control entirely otherwise. Don't ship if any auth-boundary risk surfaces. |
| O9 | 🟡 | **Hyperparam diff between runs.** | Centralize. | Small | Strip env vars / secrets before publishing. Whitelist of safe-to-show flags. |
| O10 | ✅ | **Run-level "abort reason" field.** Operator's recorded reason inline next to "superseded" badge. | Reviewers need to understand WHY a run died. | Small | — |
| O11 | ✅ | **Reproducibility manifest** per ckpt: git SHA, dataset SHA, RNG state, exact CLI invocation. One-click "recreate this run." | The single most powerful lab-notebook feature. | Medium | — |
| O12 | ✅ | **Live GPU memory chart over time** (30-min rolling). | Catch leaks before disaster. | Small | — |

## Community perspective (visitors, potential users, contributors)

| # | Verdict | Idea | Why | Effort | Constraints / notes |
|---|---|---|---|---|---|
| C1 | 💬 | **"Try it" demo widget.** Drag image → in-browser ONNX/WebGPU model → side-by-side. | "Show me, don't tell me." | Big | User: prefer v5 or v6.1 (NOT v4). Worried about cost. Q: do we self-host inference (cheap once cached) or run client-side WebGPU (free + stays in browser)? |
| C2 | ✅ | **Architecture diagram, interactive.** Click any block → drawer with spec memo + code links. | Project's main differentiator. | Medium | — |
| C3 | 🟡 | **"Subscribe" — email/RSS/Matrix/Discord** for milestones. | Captures interest at peak attention. | Medium | MUST be free + low-maintenance. RSS = static file, no infra. Email = needs SMTP service. Recommend: RSS + Atom feed first, no email. |
| C4 | ✅ | **Progress bar to convergence.** "X / 100K steps · Y% · ETA Z hr". | Makes run feel in-progress. | Small | — |
| C5 | ✅ | **Citation block.** Auto `.bib` entry + "how to cite" sentence. | Researchers reference work. | Small | — |
| C6 | ✅ | **Glossary popovers** for jargon (HAT, GS-STVSR, EWA, Net2Net, etc.). | Lower the entry barrier. | Small | — |
| C7 | ✅ | **"What changed since your last visit"** via localStorage. | Repeat-visitor engagement. | Small | — |
| C8 | 💬 + 🆕 | **Latency/throughput per GPU class.** "v6.1 at 1080p→4K on RTX 3080 Ti = X ms · M1 Max = Y · Steam Deck = Z." | The latency claim gets vendors interested. | Big | **🆕 User suggests writing custom NVIDIA kernel FIRST so we can measure during training.** Big architectural question — see "Open architectural questions" below. |
| C9 | 🟡 | **Discord/Matrix/IRC link.** | Lower question-asking barrier. | Small | We don't have a Discord yet. Decide: create one OR link Matrix room (lower maintenance) OR link a github discussions tab (free + already integrated). |
| C10 | 💬 + 🆕 | **Public roadmap card.** "Now / Next / Later / Wishlist." Versioned. | Visitors want "where is this going?" | Small | **User explicitly requested interactive Q&A to hash out the roadmap content before shipping.** See "Open architectural questions" below. |
| C11 | ✅ | **Embed-friendly chart cards.** Per-chart "Embed" button → iframe snippet. | Distribution / backlinks. | Small | — |
| C12 | ✅ | **One-click "open in viewer"** for held-out frames. | Visitors pixel-peep without Python. | Small | User noted: clarify it never required Python. Pure JS viewer is the model, just needs deep-linkable URL. |

## Researcher perspective (ML peers, academic reviewers)

User direction for ALL R-rows: "make it BEAUTIFUL, easy to use, exactly as a scientist would want." Apply to every researcher-facing addition.

| # | Verdict | Idea | Why | Effort | Constraints / notes |
|---|---|---|---|---|---|
| R1 | ✅✅✅ | **Pareto frontier 2D scatter** (PSNR x LPIPS, model = point, training trajectory = path). | Universally recognized ML-paper figure. | Medium | — |
| R2 | ✅✅✅ | **Per-frame breakdown.** Histogram of per-frame deltas + win/loss matrix vs bicubic. | Reveals "every frame" vs "averaged" wins. | Medium | Beautiful + scientist-grade UI required. |
| R3 | ✅ | **Confidence intervals.** Wilson 95% CI on "beats bicubic" %, std-dev / IQR on means. | Single-number means lie. | Small | — |
| R4 | ✅ + 🆕 + 🆕 | **Wall-clock training time + GPU-hours + kWh + USD.** | "PSNR per joule / per dollar" comparison. | Medium | **🆕 Add Contribute / Contact / Sponsor button** at every cost-breakdown panel. **🆕 Add cost projection: USD to reach DLSS-4-equivalent quality on rented cloud GPUs** (B200 / H100 / A100 / 4090 instance pricing × estimated GPU-hours). |
| R5 | ✅ | **Model card.** Param count, FLOPs, peak memory, license, training data composition. | Institutional gating. | Small | — |
| R6 | ✅ | **Out-of-distribution detection metric.** Per-scene OOD-confidence on UE5 / Unity / Source 2 / TartanAir. | Defends against "fails on unseen content" critique. | Medium | — |
| R7 | ✅ | **Failure-mode catalog.** Grid of frames where v6.1 was WORST. | Credibility. | Medium | — |
| R8 | ✅ | **Loss-decomposition normalization toggle.** Absolute vs fraction-of-total. | Different question, different answer. | Small | — |
| R9 | 🟡 | **Direct A/B test viewer.** Twin-pane image-slider on a viz frame. | Visual evidence. | Medium | UI-density caveat: don't make it busy. If toggles+views compromise usability, can it. |
| R10 | ✅ + 🆕 | **Spectral / FFT analysis.** | High-frequency detail check. | Medium | **🆕 Cross-model comparison** — show OSS spectrum next to bicubic, DLSS, FSR, XeSS spectra (where measured). |
| R11 | ✅ | **Per-region loss heatmap.** Pixel-wise error visualization on a representative frame. | Spatial breakdown — rooftops vs sky. | Medium | — |
| R12 | ✅✅✅✅✅ | **Linkable URL state.** `?run=v6.1&step=5000&chart=loss-decomp&zoom=1500-3000` | Citeable views. | Small | — |

## Cross-cutting / infrastructure

| # | Verdict | Idea | Why | Effort | Constraints / notes |
|---|---|---|---|---|---|
| X1 | 🟡 | **Public dataset preview.** Carousel of TartanAir oldtown frames. | "Held-out batch" → concrete content. | Small | Constraint: only if R2 storage cost stays minimal. ~64 frames × ~500KB = ~32MB. Trivial. |
| X2 | ✅ | **Schema versioning** in data.json. | Future-proof for third-party readers. | Small | — |
| X3 | ✅ + 🆕 | **Multi-run live monitoring** + cloud-GPU compatibility. | Cascade plan (Pico → Standard → Heavy) is multi-run. | Medium | **🆕 Add "How can you help?" button prominently.** Surfaces requests for: financial sponsorships, donations, GPU-time, lended GPUs, gifted GPUs (high-VRAM), and more. |
| X4 | 💬 | **API access.** `/api/v1/runs/<id>/score_log`. | Tools/blog cards/aggregations. | Small | User concerned about cost + becoming-a-problem. Q: rate-limit + Cloudflare cache + Bearer-optional read-only. R2 egress is free; throughput limit at the Worker edge. |
| X5 | ✅ | **Status page.** Trainer / watcher / Worker / R2 / DNS health. | Self-monitoring. | Small | Tooltips for what each service does for training. |
| X6 | 💬 | **Audit log.** Every Worker write logged (key + size + timestamp + token-id). | Forensics if secret leaks. | Medium | User unsure (public dashboard concern). Q: log to private R2 path the dashboard never reads, surface only count + last-write-time publicly. |
| X7 | ✅ | **Dataset transparency card.** What's in TartanAir Easy / oldtown held-out / future cross-engine sets. | Reviewers want training-data composition. | Medium | Make synergistic with R5 model card — link both. |

## Open architectural questions (need conversation)

These are the items where the user explicitly requested interactive Q&A before scoping:

1. **C10 — Public roadmap content.** Hash out Now / Next / Later / Wishlist with the user.
2. **C8 / 🆕 custom NVIDIA kernel.** Should the custom kernel be written FIRST so cross-vendor benchmarks are meaningful? Implies a multi-week detour from the v6.1 → v6.2 → Standard model cascade.
3. **C1 — Try-it demo cost model.** Self-host inference (cheap once cached) vs client-side WebGPU (free, runs in browser, slower)?
4. **O5 — Ckpt download cost / publication gate.** Auto-publish all ckpts to R2 (storage cost) vs operator-blessed-only with a "publish" button?
5. **O6 — Cost meter usefulness.** Worth showing or noise — only ship if it pairs with R4 cost projection?
6. **X4 — API access.** Throttle/cost shape OK? Bearer-required vs anonymous-rate-limited?
7. **X6 — Audit log.** Public visibility limits (count-only vs full log)?

## Honest priority recommendation (pick one to dispatch first)

After triage, the YES rows that combine **highest leverage × smallest effort**:

1. **R12** Linkable URL state (5×YES) — easy, every other feature benefits from citeable URLs
2. **R1** Pareto scatter (3×YES) — universally-recognized ML figure
3. **R2** Per-frame breakdown (3×YES) — unique credibility differentiator
4. **C7** "What changed since last visit" — small, drives repeat engagement
5. **C2** Architecture diagram interactive — project's main differentiator made explorable

## Status tracking

Backlog state. Pick a row → spawn a codex prompt at `/tmp/prompt-dashboard-<id>.txt` → dispatch via `scripts/dispatch_codex.sh`. Mark the row "shipped (commit `<sha>`)" once landed.
