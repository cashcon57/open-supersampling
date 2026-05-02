# OSS-Gaussian — Paper Drafts

Holding pen for paper drafts and the BibTeX library backing the architecture docs and experiment memos.

**Do not draft a full paper until a result clears one of the gates in `gaussian-network-architecture.md` §7 (V0 → V2 staging).** Premature drafts produce weak papers; weak papers in this space hurt future stronger papers.

## Layout

| Path | Purpose |
|---|---|
| `oss-gaussian-v0-draft.md` | Workshop-paper-shaped draft. Stays in skeletal form until V0 clears its bicubic gate. |
| `oss-gaussian.bib` | Single BibTeX file shared across drafts and architecture docs. |
| `figures/` | All figures (PNG/SVG/PDF). Generated from training runs; never edit by hand. |
| `experiments-index.md` | Auto-curated index of `docs/superpowers/experiments/*.md` entries grouped by topic and date. Update whenever a new experiment memo lands. |

## Drafting cadence

1. **Per training run:** add an entry to `docs/superpowers/experiments/YYYY-MM-DD-<topic>.md` (hypothesis → setup → result table → decision). Cite that path in the commit message.
2. **Per architectural change:** update `gaussian-network-architecture.md` and add the citing paper to `oss-gaussian.bib`.
3. **At V0 gate clear:** promote the strongest experiment memos into `oss-gaussian-v0-draft.md`'s Results section. The memos are the paper-in-pieces; the draft is the assembly.
4. **At V1 gate clear:** branch to `oss-gaussian-v1-draft.md`; V0 draft remains as a workshop submission.

## What goes in `oss-gaussian.bib`

- Every paper cited in `docs/superpowers/research-synthesis-*.md`.
- Every paper cited in `gaussian-network-architecture.md` §8.
- Every paper that informed a design decision in an experiment memo.

Keep the bib in **`@{key, ... }` BibTeX**, not BibLaTeX, so any LaTeX engine ingests it.
