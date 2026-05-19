# Starting up after wipe

Recovery runbook for the **3080 Ti training host** after a clean OS reinstall (Windows 11 → CachyOS / Arch Linux native, 2026-05-17 wipe window).

The goal: get the v7-pico-005 training run resumed on Linux from the last preserved checkpoint, restore home-lab containers, and reach a state where development on this repository continues uninterrupted.

This document assumes:
- The new OS is **CachyOS** (Arch-based, KDE Plasma) or any modern Arch-based Linux on the same hardware (Ryzen 7 5800X3D + RTX 3080 Ti, 12 GB VRAM).
- Network is up and the user can reach `github.com`, `huggingface.co`, `download.pytorch.org`, and Cloudflare R2.
- The user has out-of-band access to the **age private key** that decrypts R2-backed secrets (see "Secrets restore" below).

> **TL;DR**: install nvidia drivers + docker + nvidia-container-toolkit, clone the repo, decrypt `.secrets/` from R2, re-download TartanAir, drop a flash-attn wheel into `docker/trainer/`, then `docker compose up -d` from `docker/trainer/`. The trainer auto-resumes from `archive/v7-pico-005-snapshot-2026-05-16/step-00005000.pt` once that file is in the host's `~/checkpoints/srcnn-v7.0-pico-005/` directory.

---

## 0. Pre-wipe checklist

Run these **before** booting the install media. Any step skipped is permanently lost.

- [ ] **R2 secrets backup ran today.** Verify in dashboard or with `wrangler r2 object list oss-secrets-backup --remote | head`. The latest archive should be from the same day as the wipe. Run `bash scripts/backup-secrets.sh` from the Mac if unsure.
- [ ] **Age private key is mirrored off this host.** The key lives at `/Users/cashconway/OpenSuperSampling/.secrets/age-secret-key.txt` on the Mac. Confirm `md5 -q` matches at least one other mirror (`g14`, paper printout, or password-manager attachment). See `.secrets/RECOVERY-README.md` for the mirror matrix.
- [ ] **`.secrets/` rsynced to the homelab laptop.** When `g14` is back online, run:
  ```bash
  rsync -avz --delete \
    /Users/cashconway/OpenSuperSampling/.secrets/ \
    cashc@g14:~/OpenSuperSampling-backup/secrets/
  ```
  If `g14` is offline at wipe time, the R2 backup is the only copy. Restore via that path after the wipe and copy a fresh mirror to `g14` once it returns.
- [ ] **Home-lab docker-compose files are backed up.** These are not in this repository. The other agent (per 2026-05-18 conversation) is preserving them; confirm with that workstream before pulling the trigger.
- [ ] **Cloudflare Tunnel credentials are noted.** The tunnel ID and token live in the cloudflared container config. After wipe, the tunnel must be reauthorized either by reusing the same UUID + token or by creating a new tunnel and updating DNS routes.
- [ ] **Tailscale auth keys** in `.secrets/` or the Tailscale admin console. Re-auth on the new install is one command if you keep the same node identity; otherwise a fresh `tailscale up --authkey=...` is required.
- [ ] **The training run that was active at wipe time is at step 5000.** History row count: 47. Loss: `sr_charbonnier=0.0063, sr_lpips=0.113, total=0.120, canvas_count=2304`. See `archive/v7-pico-005-snapshot-2026-05-16/history.jsonl` for the full trajectory and `archive/v7-pico-005-snapshot-2026-05-16/step-00005000.pt` for the resume-able weights.

---

## 1. OS install

Out of scope for this document — assume the user has installed CachyOS or equivalent and can sudo. Hardware:

| Component | Spec |
|---|---|
| CPU | AMD Ryzen 7 5800X3D (Zen 3, sm_86-tier AVX2 max, **no AVX-512**) |
| GPU | NVIDIA GeForce RTX 3080 Ti, 12 GB GDDR6X, compute capability 8.6 |
| RAM | (User to fill in) |
| Drives | (User to fill in) |

The Zen 3 / no-AVX-512 detail matters: torch wheels 2.9+ on Windows have an AVX-512 codegen leak that crashes on this CPU. Linux wheels from the same versions are built with GCC and **do not** have this issue, so on CachyOS the upgrade path to a newer torch is open without revisiting that bug.

---

## 2. NVIDIA driver

```bash
# CachyOS has nvidia-dkms or nvidia-open-dkms in the default repo.
# Use the proprietary stack for full CUDA support.
sudo pacman -S nvidia-dkms nvidia-utils nvidia-settings

# Verify after reboot:
nvidia-smi
# expect: RTX 3080 Ti at driver >= 555.xx, CUDA Version 12.4 or newer.
```

CUDA toolkit is **not required** on the host — torch wheels bundle their own CUDA runtime, and the docker image ships its own. The host only needs the kernel driver and the nvidia user-space libraries (`libnvidia-ml.so`, etc.) that the driver package installs.

---

## 3. Docker + NVIDIA Container Toolkit

```bash
sudo pacman -S docker docker-compose nvidia-container-toolkit
sudo systemctl enable --now docker
sudo usermod -aG docker $USER
# log out and back in so the new group membership takes effect

# Configure docker to use the nvidia runtime
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Smoke test:
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
# expect: the same 3080 Ti listing as nvidia-smi on the host.
```

---

## 4. Clone the repository

```bash
mkdir -p ~/OpenSuperSampling
git clone https://github.com/cashcon57/open-supersampling.git ~/OpenSuperSampling
cd ~/OpenSuperSampling
```

The repo now (post-wipe-prep commit) contains:

| Path | Purpose |
|---|---|
| `archive/v7-pico-005-snapshot-2026-05-16/` | Active-run checkpoints + history through step 5000 |
| `archive/legacy-runs/` | History / score logs from prior runs (v4-prod, v5-validated, v6-pico, v6.1-pico-001, v6.2-pico-002) — no checkpoints, only metrics |
| `archive/legacy-windows-launchers/` | Historical reference: the Windows PowerShell launcher scripts from the wiped host. Not used on Linux. |
| `docker/trainer/` | Dockerfile + entrypoint + docker-compose.yml for the trainer container |
| `Starting up after wipe.md` | This document |

---

## 5. Secrets restore

The `.secrets/` directory is intentionally **not** in the repo (it is in `.gitignore`). After wipe, restore it from one of three sources, in priority order:

### 5a. From the homelab laptop (`g14`)

If `.secrets/` was rsynced to `g14:~/OpenSuperSampling-backup/secrets/` before the wipe:

```bash
rsync -avz cashc@g14:~/OpenSuperSampling-backup/secrets/ ~/OpenSuperSampling/.secrets/
chmod 700 ~/OpenSuperSampling/.secrets
chmod 600 ~/OpenSuperSampling/.secrets/*
```

### 5b. From R2 (the encrypted nightly backup)

If `g14` is offline or its mirror is stale, decrypt the latest R2 archive. You need the **age private key** (one ~74-character line, starts with `AGE-SECRET-KEY-1`). Keep this off-host: paper printout, password-manager attachment, or a USB stick.

```bash
sudo pacman -S age
npm install -g wrangler   # or use the docker image cloudflare/wrangler

export CLOUDFLARE_ACCOUNT_ID=c067afd6ea60a95b946c63c599095a65
# Either restore the wrangler-authenticated session if you have it, or:
export CLOUDFLARE_API_TOKEN=<paste-fresh-token-from-dash.cloudflare.com>

wrangler r2 object list oss-secrets-backup --remote | head
# pick the newest secrets-cashs-macbook-pro-<TIMESTAMP>.tar.gz.age

wrangler r2 object get oss-secrets-backup/secrets-cashs-macbook-pro-<TIMESTAMP>.tar.gz.age \
  --file=/tmp/restore.tar.gz.age --remote

# Place the age private key at a known path then decrypt:
age -d -i ~/age-secret-key.txt -o /tmp/restore.tar.gz /tmp/restore.tar.gz.age
mkdir -p ~/OpenSuperSampling/.secrets
tar -xzf /tmp/restore.tar.gz -C ~/OpenSuperSampling/.secrets
chmod 700 ~/OpenSuperSampling/.secrets
chmod 600 ~/OpenSuperSampling/.secrets/*

# The age private key itself is NOT in the tar (the backup script excludes it).
# Place a copy at ~/OpenSuperSampling/.secrets/age-secret-key.txt manually.
```

### 5c. After restore: re-mirror to g14

Once everything works on the new host, push `.secrets/` back to `g14` so the mirror stays current:

```bash
rsync -avz --delete ~/OpenSuperSampling/.secrets/ cashc@g14:~/OpenSuperSampling-backup/secrets/
```

And re-verify the age private key hash matches at least one other mirror, per `.secrets/RECOVERY-README.md`.

---

## 6. TartanAir dataset

The dataset is **553 GB / ~550,000 small files** and is **not** preserved across wipes. It will need to be re-downloaded.

### Download

TartanAir is hosted on Microsoft Azure / public CDN. The original distributor is the TartanAir project at <https://theairlab.org/tartanair-dataset/>. Their helper script downloads per-scene per-mode .zip archives.

Estimated wall time on a 1 Gbit residential connection:
- **Download**: 553 GB at 100-200 MB/s = 1-2 hours
- **Extract** (50k+ zip files): 1-2 hours of disk I/O
- **Filesystem walk to populate page cache** on first trainer launch: 10-20 minutes (one-time)

Target location: `~/datasets/tartanair_extracted/`. The docker compose file looks for it there by default. Adjust by setting `OSS_DATASETS` before `docker compose up`.

### Smoke training without the full dataset

If you want to validate the GPU + container stack before the full download finishes, the trainer accepts `--max-triplets N` to cap the dataset size. Override the compose entrypoint:

```bash
docker compose run --rm oss-trainer \
  python -u scripts/sr_train_v7.py \
    --tartanair-root /datasets/tartanair_extracted \
    --output-dir /checkpoints/srcnn-v7.0-pico-005 \
    --steps 5005 --batch-size 1 --device cuda --log-every 1 \
    --backbone-kind hat_tiny --canvas-capacity 1024 --max-hr-crop 192 \
    --no-compile --max-triplets 32
```

5 steps in ~30 seconds confirms the stack is healthy. The full run can resume later from the same checkpoint.

---

## 7. Flash-attn 2 wheel

The trainer's HAT-tiny backbone routes attention through `flash_attn_func` when the package is installed (see `oss/sr/v6/hat.py:42`). The Dockerfile expects a pre-built Linux wheel to be in the build context as `flash_attn-*.whl` to avoid a 10-15 minute source compile that has been known to OOM-kill dockerd under WSL2 (which the new Linux native install will not have, but the same RAM pressure can still bite on a 32 GB host).

### Preferred: official Dao-AILab Linux wheel

```bash
cd ~/OpenSuperSampling/docker/trainer
# Match torch and CUDA versions. For torch 2.4.1 + cu124 + cp311:
curl -L -o flash_attn-2.7.0.post2-cp311-cp311-linux_x86_64.whl \
  https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.0.post2/flash_attn-2.7.0.post2+cu12torch2.4cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
# (verify the exact asset URL on the release page; the cxx11abi suffix
# matches the cxx11abi value of the torch wheel bundled in the
# pytorch/pytorch:2.4.1-cuda12.4-cudnn9-devel base image)
```

### Fallback: build from source inside a temporary container

If no matching wheel exists, run a one-shot container that builds the wheel with `MAX_JOBS=2` (capped parallelism to avoid OOM) and copies the resulting wheel out:

```bash
cd ~/OpenSuperSampling/docker/trainer
docker run --rm -v $(pwd):/out \
  pytorch/pytorch:2.4.1-cuda12.4-cudnn9-devel \
  bash -c "pip install --no-cache-dir packaging ninja wheel setuptools && \
           MAX_JOBS=2 pip wheel flash-attn==2.7.0.post2 --no-build-isolation -w /out"
# Result: /out/flash_attn-2.7.0.post2-cp311-cp311-linux_x86_64.whl
```

---

## 8. Place checkpoints

```bash
mkdir -p ~/checkpoints/srcnn-v7.0-pico-005
cp ~/OpenSuperSampling/archive/v7-pico-005-snapshot-2026-05-16/*.pt \
   ~/checkpoints/srcnn-v7.0-pico-005/
cp ~/OpenSuperSampling/archive/v7-pico-005-snapshot-2026-05-16/history.jsonl \
   ~/checkpoints/srcnn-v7.0-pico-005/
cp ~/OpenSuperSampling/archive/v7-pico-005-snapshot-2026-05-16/score_log_v7.json \
   ~/checkpoints/srcnn-v7.0-pico-005/
```

The trainer's auto-resume logic finds `step-NNNNNNNN.pt` files in `--output-dir`, picks the highest-numbered one, and resumes both model and optimizer state. Confirm it sees the right file with:

```bash
ls -la ~/checkpoints/srcnn-v7.0-pico-005/step-*.pt
# expect step-00005000.pt as the highest.
```

---

## 9. Build and start the trainer

```bash
cd ~/OpenSuperSampling/docker/trainer
docker compose build
docker compose up -d
docker compose logs -f
```

What to look for in the logs (in this order):

1. `[entrypoint] HEAD: <hash> <commit message>`
2. `[train] dataset: 63633 triplets` — dataset enumerated from `/datasets/tartanair_extracted`
3. `[train] resume <- /checkpoints/srcnn-v7.0-pico-005/step-00005000.pt`
4. `[train] resumed at step 5000`
5. Roughly 5-10 minutes later: `[step  5050] loss=... sr_char=... canvas=... elapsed=...`

If step 4 prints but step 5 never does, the trainer is most likely stuck in dataset I/O or first-batch warmup. Wait at least 20 minutes (the Dockerfile healthcheck's `start-period`) before suspecting a hang.

If step 4 prints a step number lower than 5000, the wrong checkpoint was picked up. Recheck the `~/checkpoints/srcnn-v7.0-pico-005/` directory.

### Expected per-step times on Linux + flash-attn + bf16 (on a 3080 Ti)

| Canvas count | Expected s/step |
|---|---|
| 1024 (warmup, steps 0-500) | 7-9 |
| 2304 (post-materialization, current state) | 18-25 |
| 4096+ (later in training) | 30-50 |

This is approximate. The Windows-side reference at step 4400 was ~29 s/step at canvas=2304, and Linux native + proper flash-attn linkage is expected to shave 20-30% off that.

---

## 10. Dashboard publisher

The public dashboard at <https://opensupersampling.org/> is served by the Cloudflare Worker `oss-dashboard-uploader` and an R2 bucket. The publisher script that uploads `data.json` and viz frames lives at `scripts/watch_and_publish.sh` and was previously running on the wiped host.

To resume it on the new Linux host:

```bash
# WORKER_UPLOAD_URL and SHARED_SECRET come from .secrets/
# (see r2-credentials.env or cloudflare-master-token.env).
export WORKER_UPLOAD_URL=https://upload.opensupersampling.com
export SHARED_SECRET=$(grep CF_UPLOAD_SECRET ~/OpenSuperSampling/.secrets/cloudflare-master-token.env | cut -d= -f2)

# Run as a systemd-system service for stability. A reference unit:
sudo tee /etc/systemd/system/oss-dashboard-publisher.service > /dev/null <<'UNIT'
[Unit]
Description=OSS Dashboard Publisher (watch checkpoints + upload to R2)
After=network-online.target docker.service

[Service]
Type=simple
User=$USER
WorkingDirectory=/home/$USER/OpenSuperSampling
Environment=WORKER_UPLOAD_URL=https://upload.opensupersampling.com
EnvironmentFile=/home/$USER/OpenSuperSampling/.secrets/r2-credentials.env
ExecStart=/bin/bash /home/$USER/OpenSuperSampling/scripts/watch_and_publish.sh
Restart=on-failure
RestartSec=15

[Install]
WantedBy=multi-user.target
UNIT
# replace $USER literally in that heredoc before sudo systemctl daemon-reload
sudo systemctl daemon-reload
sudo systemctl enable --now oss-dashboard-publisher.service
```

Verify the live site reflects training progress within ~30 seconds of the next `[step ...]` log line in the trainer.

---

## 11. Cloudflare Tunnel restore

The tunnel previously routed `dash.corkscrewmodmanager.com` and other ingresses through a `cloudflared` container. After wipe:

1. Re-create the `cloudflared` container with the same tunnel UUID and token (lives in the home-lab compose files, which the other agent is preserving — coordinate with them).
2. **Critical fix carried forward from the 2026-05-18 forensics**: do **not** route ingress origins at the host's Tailscale IP (`100.81.53.79:NNNN`) from inside `cloudflared`'s Docker network. The Docker bridge cannot always reach Tailscale IPs after restarts. Use `host.docker.internal:NNNN` or move `cloudflared` to `network_mode: host`.
3. Verify with `docker exec cloudflared sh -c 'nc -vz 100.81.53.79 4321'` (or equivalent) before relying on browser tests — unauthenticated 302 from Cloudflare Access only proves the edge is alive, not the origin.

---

## 12. Tailscale

```bash
sudo pacman -S tailscale
sudo systemctl enable --now tailscaled
sudo tailscale up --hostname=3080ti-linux
# follow the URL printed to authorize the node in the tailnet admin console
```

If you want to keep the existing node identity (same Tailscale IP), reuse the previous machine key from `.secrets/`. Otherwise a fresh authorization is fine and the rest of the tailnet will see the new node name.

---

## 13. Resume any in-flight conversations

The repository CLAUDE.md (root level if present) plus the user memory store at `~/.claude/projects/-Users-cashconway-OpenSuperSampling/memory/MEMORY.md` describe project conventions, ongoing work, and known-good ways to interact with the system. After the wipe, re-anchor by reading both before asking the agent to resume autonomy.

Useful pointer files in this repository:

- `RESEARCH.md` — architecture notes for v7 and the surrounding history.
- `CHANGELOG.md` — chronological log of decisions and fixes.
- `docs/coordination/2026-05-14-dashboard-layout-audit-verification.md` — the most recent layout proposal still partially un-executed.
- `archive/v7-pico-005-snapshot-2026-05-16/history.jsonl` — the loss trajectory through step 5000 against which any new training metrics should be compared (regression detection).

---

## 14. Sanity check the live dashboard

Open <https://opensupersampling.org/> after the publisher service has run for at least one cycle. The hero block should read something like:

> `v7 is training · step 5050 / 100,000 · loss <new> · viz @ step <new>`

If the step number stays stuck at 5000, the publisher is not seeing fresh writes. Common causes:
1. The publisher is reading from the wrong checkpoint directory.
2. The trainer is not actually progressing (see step 9 above).
3. The R2 upload is failing — check the publisher's stderr.

The `.com` apex now 301-redirects to `.org`. If `.com` shows a direct 200, the worker may have been redeployed without the redirect block at the top of `fetch()`. The worker source-of-truth lives at `scripts/oss_dashboard_uploader_worker.js`.

---

## 15. Final cleanup

Once the run is back up:

- [ ] Re-enable nightly secrets backup via cron / systemd timer (or launchd on the Mac, which still owns this).
- [ ] Update the age-key mirror matrix to reflect the new host. Per `.secrets/RECOVERY-README.md`, the previous mirrors were `Mac`, `3080ti-windows`, and `g14`. The wiped host's mirror is gone — replace with the new `3080ti-linux` mirror once the OS is stable.
- [ ] Update `tailnet_3080ti` memory entry to reflect the new SSH alias and any IP changes.
- [ ] Update `reference_g14_cuda_dev_host` memory entry if g14's role changes during this transition.
- [ ] Delete the `archive/` directory from the next regular branch once the new host is stable for at least a week. The data is preserved in git history and the live dashboard.

---

## Appendix A: What was on the wiped host

| Path | Preserved? | Where |
|---|---|---|
| `E:\checkpoints\srcnn-v7.0-pico-005\step-*.pt` | Yes | `archive/v7-pico-005-snapshot-2026-05-16/` in this repo |
| `E:\checkpoints\srcnn-v7.0-pico-005\history.jsonl` | Yes | same |
| `E:\checkpoints\srcnn-v7.0-pico-005\score_log_v7.json` | Yes | same |
| `E:\checkpoints\srcnn-v7.0-pico-005\viz\*.png` | No (committed copies) | Live on R2 at `opensupersampling.org/runs/srcnn-v7.0-pico-005/viz/` |
| `E:\checkpoints\srcnn-v6.2-pico-002\` (31.9 GB) | Metrics only | `archive/legacy-runs/srcnn-v6.2-pico-002/` |
| `E:\checkpoints\srcnn-v6.1-pico-001\` (1.17 GB) | Metrics only | `archive/legacy-runs/srcnn-v6.1-pico-001/` |
| `E:\checkpoints\srcnn-v5-pixel-temporal-validated\` (687 MB) | Metrics only | `archive/legacy-runs/srcnn-v5-pixel-temporal-validated/` |
| `E:\checkpoints\srcnn-prod-v4-lpips\` (259 MB) | Metrics only | `archive/legacy-runs/srcnn-prod-v4-lpips/` |
| `E:\datasets\tartanair_extracted\` (553 GB) | No (re-download) | See section 6 |
| `~/docker-oss-trainer/Dockerfile` (WSL) | Yes (reconstructed) | `docker/trainer/Dockerfile` |
| `~/docker-oss-trainer/docker_launch.sh` (WSL) | Yes (reconstructed) | `docker/trainer/docker_launch.sh` |
| `C:\Users\cashc\*.ps1` (Windows launchers) | Reference only | `archive/legacy-windows-launchers/` |
| `oss-trainer:latest` docker image (~21 GB) | No | Rebuildable from `docker/trainer/` |
| `.secrets/` directory | Yes (R2 + Mac + g14) | See section 5 |
| Home-lab docker compose files | Out of band | Coordinate with the other agent |

## Appendix B: Verification commands

After the new host is up and the trainer is running:

```bash
# 1. Container alive + healthy
docker compose -f ~/OpenSuperSampling/docker/trainer/docker-compose.yml ps
# expect: STATUS = "Up X minutes (healthy)" after start-period elapses

# 2. GPU visible inside container
docker compose -f ~/OpenSuperSampling/docker/trainer/docker-compose.yml exec oss-trainer \
  python -c "import torch; print('cuda:', torch.cuda.is_available(), 'count:', torch.cuda.device_count())"
# expect: cuda: True count: 1

# 3. Flash-attn loaded (the hat.py path)
docker compose -f ~/OpenSuperSampling/docker/trainer/docker-compose.yml exec oss-trainer \
  python -c "from oss.sr.v6.hat import _HAS_FLASH_ATTN; print('flash-attn:', _HAS_FLASH_ATTN)"
# expect: flash-attn: True

# 4. Latest step beyond 5000
tail -n 1 ~/checkpoints/srcnn-v7.0-pico-005/history.jsonl | python3 -c "import json,sys; print(json.loads(sys.stdin.read())['step'])"
# expect: a number > 5000 within ~15 minutes of starting
```

## Appendix C: Pre-wipe artifact provenance

This document and the `archive/` tree were created during the 2026-05-18 wipe-prep session. The 3080 Ti host at that time was running:

- WSL2 Ubuntu 22.04.5 (kernel 6.6.87.2-microsoft-WSL2)
- Windows 11 (driver 595.79, CUDA 13.2 reported by the driver despite no toolkit install)
- `oss-trainer:latest` docker image (last built locally, ~7.5 GB content size)
- 18+ home-lab docker containers (plex, jellyfin, pihole, vaultwarden, cloudflared, etc.)
- Active training run: `srcnn-v7.0-pico-005`, last checkpoint at **step 5000**, last history row at step 5000 with `total_loss=0.11950`, canvas=2304, elapsed_s=88818.6

All commit hashes referenced are visible in `git log` of this repository. The trainer code at the time of the snapshot is at HEAD `56a9a65` ("trainer: revert DataLoader to num_workers=0 (WSL2 deadlock)"). Subsequent commits, if any, should be picked up automatically by the entrypoint's `git fetch + reset --hard origin/main` on container start.
