# Public dashboard setup

The workflow publishes a static GitHub Pages site from `dashboard-public/` on
the `gh-pages` branch. The public URL is:

https://cashcon57.github.io/open-supersampling/dashboard-public/

## Required repository secrets

Go to Repo Settings -> Secrets and variables -> Actions -> New repository
secret and add:

- `TS_OAUTH_CLIENT_ID` - Tailscale OAuth client ID
- `TS_OAUTH_SECRET` - Tailscale OAuth secret

Until both secrets exist, the GitHub Action skips the training-host fetch and
republishes whatever dashboard data is already cached on `gh-pages`.

## Tailscale OAuth client

Create the OAuth client at:

https://login.tailscale.com/admin/settings/oauth

Scopes:

- `auth_keys:write`
- `devices:read`

The workflow uses `tailscale/github-action@v3` and joins with `tag:ci`. Make
sure the tailnet ACL permits that tag to reach `3080ti-windows` over SSH/rsync.

## Training host access

Default host settings in the workflow:

- Host: `3080ti-windows`
- Remote path: `E:/checkpoints`
- Runs allow-listed for public sync:
  - `srcnn-v6.1-pico-001`
  - `srcnn-v5-pixel-temporal-validated`
  - `srcnn-v6-pico-001`

Optional repo variables if the SSH/rsync setup differs:

- `DASHBOARD_REMOTE_USER`
- `DASHBOARD_REMOTE_PATH`
- `DASHBOARD_REMOTE_SHELL`
- `DASHBOARD_REMOTE_SSH`

The workflow intentionally syncs only `metrics.json`, `score_log.json`,
`viz/step-*.png`, and the last 100 lines of `train.log`.

## GitHub Pages

Go to Repo Settings -> Pages:

- Source: Deploy from a branch
- Branch: `gh-pages`
- Folder: `/ (root)`

The workflow commits into `dashboard-public/` on that branch, so the dashboard
path is `/open-supersampling/dashboard-public/`.

## First publish

The action runs every 10 minutes, or you can run it manually from Actions ->
Dashboard snapshot -> Run workflow. After the first successful publish, this
URL is live:

https://cashcon57.github.io/open-supersampling/dashboard-public/

## Bandwidth

Each fetch is expected to transfer about 5 MB. At 144 scheduled runs per day,
that is about 720 MB/day from the training host. Sanity-check the home
connection before leaving the schedule enabled long term.
