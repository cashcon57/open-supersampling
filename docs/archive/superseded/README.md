# Superseded work, kept for reference

## `2026-06-hermes-training-status-alt.patch`

An uncommitted June 2026 change from the Hermes working clone on homelab-desktop (`~/.cache/my-github-repos/open-supersampling`, base commit `a97a180`). It adds `scripts/training_status.py` (docker/trainer health probes and a health classifier), a refactored `scripts/training_api.py`, an in-page training-controls panel for the public dashboard, and tests.

Not merged on 2026-10-22 because `main` had already moved to a different design: the public dashboard links to a control page served by `training_api.py` over Tailscale only, while this patch keeps the API secret in the public page's `localStorage`. `main`'s `training_api.py` is also the deployed one and has more features (pause/resume, scheduled resume, progress status).

The health-classification logic in `training_status.py` and its tests may be worth porting into `main`'s API later. Apply with `git apply --3way` against `a97a180`.
