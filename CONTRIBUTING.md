# Contributing to OpenSuperSampling

Thanks for considering OSS. The project is pre-alpha, solo-maintained, AI-augmented, and actively training. We accept contributions across research, engineering, and infrastructure tracks. This document is the practical guide.

## Quick links

- **Live dashboard:** [opensupersampling.com](https://opensupersampling.com) — current run + history
- **Architecture spec (v6.2):** [`docs/architecture/2026-05-08-v62-arch-v4-spec.md`](docs/architecture/2026-05-08-v62-arch-v4-spec.md)
- **Priority stack:** [`docs/research/2026-05-08-phase4-priority-stack-v4.md`](docs/research/2026-05-08-phase4-priority-stack-v4.md)
- **Validated discoveries:** [`docs/research/2026-05-08-validated-discoveries-log.md`](docs/research/2026-05-08-validated-discoveries-log.md)
- **Hypotheses (forward-looking):** [`docs/research/hypotheses/`](docs/research/hypotheses/)
- **Lab notebook:** [`docs/superpowers/experiments/`](docs/superpowers/experiments/)

## What we need most (right now)

Roughly priority-ordered:

1. **CUDA kernel engineers** for v6.2 Tier-1 kernels: low-rank R=4-8 splat with conic row recurrence, custom counting-sort tile bin, TBDR backward via shared-memory atomics. See [v4 priority stack](docs/research/2026-05-08-phase4-priority-stack-v4.md) Tier 1.
2. **GPU hardware time** on AMD RDNA3, Intel Arc, and Apple M3/M4 — for cross-vendor kernel development. Loaners or remote access both work.
3. **Game engine integrators** with UE5 / Unity HDRP / Source 2 internals knowledge. The DLL-shim runtime is designed but unbuilt; first integration is the hardest.
4. **Distillation engineers** for the HAT-Tiny → ≤1M student pipeline (NAFNet-block / 3-layer EfficientViT-lite candidates).
5. **TensorRT / ONNX Runtime / DirectML experts** for the cross-vendor inference runtimes (NVIDIA primary, others sequenced after).
6. **Vulkan / Slang shader engineers** for the No-ML variant (ships alongside v6.2; targets Steam Deck and integrated GPUs).
7. **Eval / benchmark engineers** for apples-to-apples comparison vs DLSS / FSR / XeSS — this is what unlocks the publication path.
8. **Research collaborators** — particularly anyone working on 2D Gaussian SR, neural supersampling, or canvas-based temporal rendering.

If you have a flash of "I could help with X but X isn't on the list," still reach out. Solo maintainers miss things.

## How to contribute

### Code contributions

```bash
# 1. Fork + clone
git clone https://github.com/<your-fork>/open-supersampling
cd open-supersampling

# 2. Install dev environment (Python 3.12+ recommended; CUDA optional but useful)
python -m venv venv
source venv/bin/activate
pip install -e ".[dev]"

# 3. Run the test suite (smoke check < 60s; full suite a few minutes)
pytest tests/v6/ -x -q

# 4. Create a branch + open a PR against main
git checkout -b your-feature
# ...changes...
git commit -m "feat: short description"
git push origin your-feature
# Open PR via GitHub UI
```

### Pull request guidelines

- **Tests required**: every functional change needs at least one test that would fail without your change. We have ~250 tests; please don't break them.
- **Lab-notebook discipline**: experiment-class changes (anything that affects training behavior, model quality, or reports new measurements) need a memo in `docs/superpowers/experiments/YYYY-MM-DD-<slug>.md`. Template: question, method, inputs, output, reproducibility.
- **Commit messages**: imperative mood, focused on WHY not what. Conventional Commits style preferred (`feat:`, `fix:`, `perf:`, `refactor:`, `docs:`, `test:`).
- **Avoid refactors during behavior changes**: separate "make it work" PRs from "make it pretty" PRs.
- **No emojis in code/commits/docs** unless the file already uses them. Match local style.

### Research contributions

- **Hypothesis records** live in [`docs/research/hypotheses/`](docs/research/hypotheses/). New formulations / claims go here BEFORE driving a v6.2 design decision. Status field tracks `untested` → `in-progress` → `validated` / `refuted` / `inconclusive`.
- **Lab notes** go inside the corresponding hypothesis file. Append; don't overwrite.
- **Cross-references**: when you cite a paper, add it to [`BIBLIOGRAPHY.md`](BIBLIOGRAPHY.md) with a one-sentence note on how OSS uses or relates to it.
- **Counter-evidence is welcome**: if your experiment refutes an existing claim, mark the hypothesis `refuted` and update [`docs/research/2026-05-08-validated-discoveries-log.md`](docs/research/2026-05-08-validated-discoveries-log.md). This is more valuable than a positive result.

### Documentation contributions

Documentation PRs are encouraged. Particularly:
- Improving the README (without diluting the honest-status-vocabulary discipline)
- Filling in missing module docstrings
- Writing per-script READMEs in `scripts/`
- Editing this CONTRIBUTING.md as the project evolves

### Infrastructure contributions

- Dashboard improvements (`dashboard-public/index.html` is the entry point)
- Cloudflare Worker / R2 publishing pipeline (`scripts/watch_and_publish.sh` + `scripts/3080ti/`)
- CI improvements (`.github/workflows/`)
- Codex queue runner (`scripts/codex_queue_runner.sh`)

## How decisions get made

- **Architecture decisions**: documented in `docs/architecture/`. The v6.2 spec is the current commitment.
- **Research decisions**: every novel claim goes through the hypothesis → measurement → discovery pipeline. No empirical claim ships without a memo.
- **Process decisions**: durable feedback rules live in maintainer's auto-memory; key discipline rules referenced in this CONTRIBUTING.md.
- **Disagreements with the maintainer**: open a GitHub issue. Bring data. We respect counter-evidence.

## What we don't do

- We don't accept code-quality PRs that wrap the codebase in abstractions for future hypothetical use cases. Three similar lines is better than a premature abstraction.
- We don't accept "I added error handling for this case that can't happen" PRs. Trust internal code; only validate at boundaries.
- We don't accept PRs that bypass safety checks (`--no-verify`, etc.) without strong justification.
- We don't accept changes that silently degrade training quality. If your perf optimization costs ≥0.1 dB PSNR, it needs an architectural justification or a quality toggle.

## Channels

- **GitHub Issues**: bugs, feature requests, roadmap discussion
- **GitHub PRs**: code + docs contributions
- **Email** (`cashcon57@gmail.com`): private commercial discussions, sponsorship, hardware loaner offers, hiring inquiries
- **Live dashboard**: [opensupersampling.com](https://opensupersampling.com) — passive read-only window into current training

## Code of conduct

Be respectful. Disagree with arguments, not people. Don't be a jerk about losing technical debates — the codebase is the source of truth, not your social capital. The project is small enough that a single bad-faith interaction can derail it; please don't.

## Licensing

OSS is Apache 2.0. Contributions are accepted under the same license. By submitting a PR, you confirm:
1. You have the right to contribute the code
2. You agree to license your contribution under Apache 2.0
3. You have not included third-party code without proper attribution in [`NOTICE`](NOTICE)

## Thank you

Solo-maintained projects only survive because of contributors who show up. If you read this far, you're already helping.

— Cash Conway, maintainer
