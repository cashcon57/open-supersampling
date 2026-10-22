# Research briefs (auto-generated)

The Hermes cron job "OSS Research & Status" writes one brief per run (08:00 and 20:00) after sweeping arXiv, GitHub, archive.org and vendor news. These files are its output from 2026-05-28 onward.

**Treat them as leads, not findings.** No human has reviewed them, and paper details (authors, venues, arXiv IDs, claimed numbers) are often unverified. Before a brief drives a decision, check the primary source and write a proper memo in `docs/research/` or `docs/superpowers/experiments/`.

## Files

- `YYYY-MM-DD-oss-research-brief.md`: one brief per day. Some days also have `-midday` or `-evening` runs.
- `…-desktop.md`, `…-laptop.md`, `…-*-hermes-clone.md`: on days when two Hermes instances ran (2026-05-31 to 06-03 and 2026-08-11 to 08-23, while Hermes moved from the desktop to the laptop), both versions are kept. The unsuffixed file comes from the current laptop job, or from the desktop when the laptop has none.
- `paper-log.md`: bibliography entries Hermes appended to its local `BIBLIOGRAPHY.md` copies. Unverified; promote entries to the curated `BIBLIOGRAPHY.md` only after checking them.
- `2026-06-08-research-to-implementation-backlog.md`: a backlog Hermes derived from the early briefs. The maintained version is `docs/coordination/report-derived-action-backlog.md`.

## History

Until 2026-10-22 these briefs were never committed. They sat as untracked files in four Hermes working clones (desktop and laptop, `~/.cache/my-github-repos/open-supersampling` and `~/.hermes/repos/open-supersampling`). They were gathered here during the repo consolidation; paper PDFs the job downloaded (`docs/research/papers/`) are git-ignored and stay on the laptop.
