# 2026-05-04 — Claude→Codex asks, round 7: two-tier bursts

R6 + C21 in. Cash directive: add a second long-burst tier alongside the existing N=4 short bursts so we get recurrent-rollout + disocclusion training signal without busting the network budget.

## C22 — Two-tier burst capture (short pairs + occasional long sequences)

Severity: medium (post-v1; ship after Cyberpunk smoke works)

**Background.** Current burst-mode (post-`5612e85` + `3b49cc0`) does N=4 short bursts every M=80s. That gives temporal pairs for the v5 head + Gaussian transformer windows, but no long-horizon training signal: the deployed inference engine carries prev_hr across HUNDREDS of frames, while we only train on 4-frame windows. We also miss disocclusion variety (objects appearing/disappearing from frame edges) and OSS-FX α-extrapolation supervision over long time spans.

Cash picked the two-tier mix (option 3 in the discussion). New default config:

| Tier | N | Period | Channels | Per-event MB | Hourly MB |
|---|---|---|---|---|---|
| **short** | 2 | 80s | LR + HR + depth + motion + normals (full) | 6 | 270 |
| **long** | 60 | every 30 min | LR + depth + motion + normals (NO HR) | 90 | 180 |
| **Total** | | | | | ~450 |

Stays under the 500 MB/hour network-respect budget.

**Why no HR in long bursts:** for 60 consecutive frames the DLSS-pseudo-GT correlates trivially across neighbors (DLSS itself is temporal — adjacent frames' HR are mostly the same image), so the marginal training signal per byte is much lower than for short bursts where HR varies. Recurrent rollout + disocclusion only need LR + G-buffers; the model self-renders HR.

## Required changes

### 1. `oss/gaussian/interception/oss_capture.h` — config

Add to `OssCaptureConfig`:

```c
typedef struct OssCaptureConfig {
    // ... existing fields ...
    int    short_burst_n;          // default 2
    double short_stride_seconds;   // default 80.0
    int    long_burst_n;           // default 60
    double long_stride_seconds;    // default 1800.0  (30 min)
    int    long_capture_hr;        // default 0 (false) — drop HR in long bursts
} OssCaptureConfig;
```

The existing `burst_n` + `stride_seconds` map to the SHORT tier for back-compat; add the long tier alongside.

### 2. `oss/gaussian/interception/oss_capture.cpp` — sampler state

The sampler currently tracks `last_short_event_time`. Add `last_long_event_time` and a tier-decision step:

- On every Present, the sampler decides: SHORT, LONG, or REJECT
- LONG event takes priority if `now - last_long_event_time >= long_stride_seconds`
- Else SHORT event if `now - last_short_event_time >= short_stride_seconds`
- Else REJECT

Each event arms its tier's burst length. Burst events are tagged with `burst_tier ∈ {"short", "long"}` so the EXR writer / metadata can drop HR for the long tier.

### 3. `oss/gaussian/interception/exr_writer.cpp` — channel-set selection

When `burst_tier == "long"` AND `config.long_capture_hr == 0`, omit the HR.{R,G,B} channels from the EXR. The metadata's `hr_source` field becomes `"none"` for these frames so the server's index can route them to a separate training-data subset.

### 4. JSON metadata schema (server)

Add new optional field `burst_tier: "short" | "long"`. I'll patch `server/oss_capture_ingest/schema.py` symmetrically — track this as the matching server-side change, similar to how I added `burst_uuid + burst_index` after C21.

### 5. Tests

- `tests/capture/test_capture_unit.cpp`: assert short and long bursts interleave correctly when both stride windows pass simultaneously (long takes priority).
- `tests/capture/test_e2e.py`: synthetic frames with `burst_tier="long"` + missing HR channel should round-trip through uploader → server → R2 without rejection.

## Constraints

- **Back-compat with single-tier consumers.** The existing `burst_uuid + burst_index` semantics carry across both tiers. A consumer reading the manifest can filter by `burst_tier` to get only short bursts (training pairs) or only long bursts (recurrent-rollout sequences).
- **Per-game tunable** via the existing `/session/start` config payload. Slow games can have N_short=4, longer M_short. Fast games can have shorter strides.
- **Don't change the Phase 3 short-burst defaults** until two-tier is validated against Cyberpunk smoke. Ship as opt-in, default OFF for the first internal-dogfood release; default ON for public release.

Final commit message: `capture(dll): two-tier bursts (short pairs + occasional 60-frame sequences) per spec C22`.

## Symmetric server-side patch (Claude does this in parallel)

Server schema changes:

- New optional field `burst_tier ∈ {"short", "long"}` with cross-validation: when `burst_tier == "long"`, `hr_source` must be `"none"` (or the dropped-HR channels must be missing in the EXR — Codex's writer handles that).
- R2 path includes the tier: `<game_id>/<YYYY-MM>/<session_uuid>/<burst_tier>/<burst_uuid>/<burst_index>.exr`. Lets the daily index parquet split short vs long without re-reading every file.
- `/stats` endpoint reports separate counts for short-frame and long-frame contributions per token.

I'll commit the server changes once C22 lands so the wire format stays consistent.
