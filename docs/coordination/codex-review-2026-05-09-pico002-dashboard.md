# 2026-05-09 codex review — pico-002 dashboard pass

## Summary

Reviewed commits `c324dd2`, `090fcf9`, `970d157`, `0e3b08a`, `dfb0aa7`, and `38817f9` on `origin/main`. The highest-risk issues are in the new viz image lifecycle and held-out frame pipeline: the split `IntersectionObserver` only unloads the first observed strip image, old observed images are not unregistered across rerenders, the held-out frame layout is inconsistent between writer/backfill/watcher/dashboard assumptions, and the PowerShell GPU guard is a non-atomic check-then-spawn. I did not see source-destruction or secret-leak behavior, but there are production-host blast-radius concerns around duplicate evals, partial PNG publication, and repeated full-tree scans/copies as the frame tree grows.

## Findings

1. **HIGH — `c324dd2` — `dashboard-public/index.html:5266`**: `ensureVizObserver()` returns the raw load observer after the first call, so only the first image gets registered with both the load and unload observers. The early return is:

   ```js
   if (_vizLoadObserver) return _vizLoadObserver;
   ```

   Every later `ensureVizObserver().observe(img)` in `renderVizStrip()` calls only `_vizLoadObserver.observe(img)`, so those frames load but never pass through `_vizUnloadObserver` and will not be blanked once they leave the larger unload margin. This defeats the stated memory-bound goal for nearly every viz strip image. Recommendation: store the wrapper itself in a stable variable and return that wrapper on subsequent calls, or change the early return to return `ensureVizObserver._wrapped`.

2. **HIGH — `0e3b08a` / `090fcf9` / `c324dd2` — `scripts/sr_temporal_held_out.py:869`, `scripts/3080ti/heldout-frames-backfill.ps1:61`, `dashboard-public/index.html:5795`**: the held-out frame path contract is inconsistent across producer and consumers. The eval writer sends each loader to a dataset subdir:

   ```py
   frames_subdir = (args.write_frames_to / name) if args.write_frames_to else None
   ```

   The backfill idempotency marker checks the root stream path:

   ```powershell
   $marker = "$framesDir\model\sample-000.png"
   ```

   while the dashboard player hardcodes only `tartanair`:

   ```js
   return `runs/${encodeURIComponent(run.name)}/heldout-frames/step-${stepStr}/tartanair/${stream}/${sampleStr}`;
   ```

   This means backfill will not skip already-filled current runs, and future multi-loader evals will publish Sintel frames that the dashboard cannot discover or select. Recommendation: define one explicit `heldout-frames` layout contract. Either write global samples directly under `{model,gt,bicubic,baseline}` as the commit text and watcher comment describe, or keep dataset subdirs and publish frame metadata (`dataset`, offset/count, streams, complete flag) in `score_log.json`/`data.json`; update the backfill marker and dashboard URL builder to consume that same metadata.

3. **HIGH — `090fcf9` / `970d157` — `scripts/3080ti/heldout-frames-backfill.ps1:69`, `scripts/3080ti/heldout-frames-backfill.ps1:86`, `scripts/3080ti/heldout-eval-supervisor.ps1:112`, `scripts/3080ti/heldout-eval-supervisor.ps1:138`**: the GPU contention guard is a non-atomic check-then-spawn. Both scripts can observe no `sr_temporal_held_out` process and then both call `Invoke-CimMethod` to start a new GPU-heavy eval. The supervisor also checks `$running` once before iterating all ready checkpoints, so another eval can start between checkpoint runs. Recommendation: acquire a named mutex or atomic lock file around the whole “check GPU, spawn eval, wait for completion” section, and use the same lock in supervisor and backfill.

4. **MEDIUM — `c324dd2` — `dashboard-public/index.html:5307`, `dashboard-public/index.html:5338`**: `renderVizStrip()` observes newly created images but removes previous strip contents with `host.replaceChildren()` without first unobserving the old `img[data-viz-src]` targets. Since the strip rerenders on polling and run changes, observers can retain detached image elements and their state over time. Recommendation: before replacing children, query existing observed images under `host` and call the wrapper `unobserve(img)` so both observers release the target.

5. **MEDIUM — `0e3b08a` — `scripts/watch_and_publish.sh:149`**: the held-out frame copy skip condition does not skip equal-mtime files after `cp -p`. The code is:

   ```bash
   if [[ -f "$dst" ]] && [[ "$dst" -nt "$png" || "$dst" -ef "$png" ]]; then
     continue
   fi
   ```

   `-ef` checks same inode, not same timestamp; after `cp -p`, source and destination usually have equal mtimes, so `-nt` is false and the file is copied again every publish cycle. Recommendation: use a source-newer test such as `if [[ -f "$dst" && ! "$png" -nt "$dst" ]]; then continue; fi`, with size or hash fallback if equal-mtime changed content matters.

6. **MEDIUM — `0e3b08a` — `scripts/watch_and_publish.sh:154`, `scripts/watch_and_publish.sh:203`**: the new recursive copy avoids `ARG_MAX`, but it still performs a full `find "$src_run/heldout-frames"` every watcher cycle, and `publish_changed()` then hashes the entire staging tree. At 256 PNGs per checkpoint, this becomes `O(total_frames)` work every 30 seconds even when no new frames landed. Recommendation: track per-run high-water/dirty state, scan with `find -newer` or a completion marker, and reserve a full reconcile for a slower cadence.

7. **MEDIUM — `0e3b08a` — `scripts/sr_temporal_held_out.py:416`, `scripts/watch_and_publish.sh:153`**: PNGs are written and copied directly at their final paths. The writer does:

   ```py
   Image.fromarray(arr_u8, mode="RGB").save(dest, format="PNG")
   ```

   while the watcher can concurrently copy any `*.png` it finds. That can publish truncated files if the watcher races a write, and the non-atomic `if not dest.exists()` guard for shared `gt`/`bicubic`/`baseline` streams can skip a file another eval has only partially created. Recommendation: write to a temp filename and atomically rename to `.png`; copy to a temp staging path and atomically rename there too. A per-step `.complete` marker would give the watcher a clean boundary.

8. **MEDIUM — `090fcf9` — `scripts/3080ti/heldout-frames-backfill.ps1:53`**: the backfill script says it handles checkpoints that already have a `score_log` row, but it iterates every `step-*.pt` in the active run:

   ```powershell
   $ckpts = Get-ChildItem $runDir -Filter "step-*.pt" -ErrorAction SilentlyContinue | Sort-Object Name
   ```

   This can duplicate the supervisor’s work on unscored/new checkpoints and can read a checkpoint that is old enough to exist but not yet intended for backfill. Recommendation: parse `score_log.json` and backfill only scored steps; keep the checkpoint-age guard for any path that still touches raw checkpoint files.

9. **MEDIUM — `970d157` — `scripts/3080ti/heldout-eval-supervisor.ps1:45`, `scripts/3080ti/heldout-eval-supervisor.ps1:47`**: the JSON-array parsing fix is directionally correct, but it can still miss a one-row score log in Windows PowerShell because `ConvertFrom-Json` can unwrap/enumerate a single-element JSON array into a scalar object, and the code only iterates when `$payload -is [System.Array]`. Recommendation: iterate over `@($payload)` and handle a scalar row object, or use `ConvertFrom-Json -NoEnumerate` where available.

10. **MEDIUM — `090fcf9` / `970d157` — `scripts/3080ti/heldout-frames-backfill.ps1:94`, `scripts/3080ti/heldout-eval-supervisor.ps1:149`**: CIM process-query failures fail open. Both wait loops use `Get-CimInstance ... -ErrorAction SilentlyContinue`; if the query fails, `$still` is empty and the script logs completion while the eval may still be running, allowing the next eval to start. Recommendation: use `-ErrorAction Stop` with `try/catch`, log the failure, and sleep/retry rather than treating “could not query” as “process exited”; apply the same fail-closed behavior to the initial process guard and spawn call.

11. **LOW — `38817f9` — `dashboard-public/index.html:5426`, `dashboard-public/index.html:5665`**: the bare-strip view remains tied to the initially opened `compareState.vizUrl`, while the new global/per-panel sliders update only each panel’s `_currentFile`. If bare-strip mode is enabled after moving sliders, or while moving the global slider, the full strip can show the original checkpoint instead of the selected checkpoint. Recommendation: define which panel or global step owns bare-strip mode, update a modal-level current URL when sliders move, and call `updateCompareBareStrip()` from the slider path.

## What looked good

- `stage_run_files` still derives the run allow-list from `RUN_CONFIG`, preserving the single source of truth instead of adding a second dashboard publish list.
- The held-out eval `sample_offset` arithmetic is internally consistent for merged metrics: the sample id is computed before appending, and the offset advances by the actual number of samples produced per loader.
- The eval loop checks the per-loader sample cap before writing frames, so partial final batches should not emit extra PNGs.
- The compare modal’s per-panel listeners are attached to fresh DOM and removed with the panel DOM on close; global slider/sync handlers are bound once and query the current grid, so I did not see listener accumulation in that path.
- The JSON parsing change in `StepsAlreadyEvaluated` fixes the main previous bug for normal multi-row JSON-array score logs.
- `dfb0aa7` is a scoped CSS cap on compare-panel height; I did not find an architecture or lifecycle regression in that commit.

## Follow-up review (commits c895330, 9e81801)

### HIGH fixes verified

- `ensureVizObserver()` fix verified: `dashboard-public/index.html:5266` now caches `_vizObserverWrapped`, `dashboard-public/index.html:5272` returns that wrapper on every call, and the wrapper still registers each image with both load/unload observers at `dashboard-public/index.html:5291`. This fixes the previous “only the first image gets unload-observed” HIGH.
- Held-out frame path fix verified for the current TartanAir run contract: the writer still writes under the per-loader directory at `scripts/sr_temporal_held_out.py:869`, the backfill marker now checks `step-NNNNNNNN\tartanair\model\sample-000.png` at `scripts/3080ti/heldout-frames-backfill.ps1:122`, the supervisor/backfill pass the step root at `scripts/3080ti/heldout-eval-supervisor.ps1:155` and `scripts/3080ti/heldout-frames-backfill.ps1:148`, and the dashboard builds `.../step-NNNNNNNN/tartanair/<stream>/sample-NNN.png` at `dashboard-public/index.html:5801` and `dashboard-public/index.html:5814`. Multi-loader selection is still future work, but the pass-1 inconsistency is fixed for the active layout.
- GPU contention fix verified for the supervisor/backfill path: both scripts use the same `C:\temp\oss-heldout-eval.lock` at `scripts/3080ti/heldout-eval-supervisor.ps1:74` and `scripts/3080ti/heldout-frames-backfill.ps1:49`, acquire it with exclusive `New-Item` at `scripts/3080ti/heldout-eval-supervisor.ps1:89` / `scripts/3080ti/heldout-frames-backfill.ps1:61`, and hold it across spawn-and-wait at `scripts/3080ti/heldout-eval-supervisor.ps1:140` and `scripts/3080ti/heldout-frames-backfill.ps1:128`. This closes the original non-atomic check-then-spawn race between those two scripts.

### New findings

- **HIGH — `9e81801` — `dashboard-public/index.html:5824`, `dashboard-public/index.html:5840`, `dashboard-public/index.html:5899`**: scored steps without backfilled frame PNGs do not get a sensible “no frames yet” state. `_heldoutResolveSteps()` treats every `score_log` row as playable, `openHeldoutVideoModal()` immediately renders the first frame, and `_heldoutScheduleTick()` starts a 12 FPS interval. For an old scored step whose R2 frame tree is still missing, one open modal continuously rewrites four `<img>.src` values and can generate sustained 404 traffic while showing broken/blank panels. Recommendation: gate playback on explicit frame metadata or a one-time `sample-000.png` probe; on `img.onerror`, stop playback for that step and show “frames not published yet” instead of continuing the interval.
- **HIGH — score-log writers — `scripts/sr_temporal_held_out.py:678`, `scripts/sr_temporal_held_out.py:691`, `scripts/sr_train_temporal.py:563`, `scripts/sr_train_temporal.py:595`, `scripts/sr_train_gaussian_temporal.py:632`, `scripts/sr_train_gaussian_temporal.py:666`, `scripts/watch_and_publish.sh:124`**: `score_log.json` still has uncoordinated writer paths outside the new GPU mutex. The held-out evaluator does a read/modify/write through a fixed `.tmp` path, while the older temporal trainers rehydrate `score_log` once at resume and then truncate/rewrite `score_log.json` from that stale in-memory list on every metrics dump. If either legacy trainer runs against a dashboard-published output dir while held-out eval appends rows, it can drop fresh held-out rows or expose partial JSON to the watcher. Recommendation: remove trainer-side `score_log.json` writes for these paths, or centralize all score-log updates behind one lock plus unique temp files and atomic replace; the watcher should only publish completed JSON.
- **MEDIUM — `9e81801` — `dashboard-public/index.html:5851`, `dashboard-public/index.html:5874`, `dashboard-public/index.html:5955`, `dashboard-public/index.html:5971`**: modal close clears the interval, but it does not release the last four held-out images. The grid remains populated, each `<img>` keeps its final `src`, and `heldoutPlayer.imgs`, `heldoutPlayer.run`, and `heldoutPlayer.steps` retain references until the next open or page unload. Recommendation: on all close/cancel paths, clear the interval, blank each image `src`, `grid.replaceChildren()`, and reset the retained player fields.
- **MEDIUM — `9e81801` — `dashboard-public/index.html:5814`, `dashboard-public/index.html:6866`, `scripts/watch_and_publish.sh:147`**: frame URL construction uses `encodeURIComponent(run.name)` while publishing mirrors raw filesystem path segments. This is fine for today’s slash-free run names and matches `runUrl()`, but future names containing `/`, `%2F`, `?`, `#`, or other path-significant characters can diverge between local directory layout, R2 object keys, and browser/server decoding. Recommendation: introduce one canonical storage slug in published metadata and keep display name separate from URL path identity.

### What still looks good

- The document-level held-out click delegation is narrow: it only matches `[data-heldout-video-run]`, does not stop propagation, and the current button at `dashboard-public/index.html:6866` is not inside a dropdown, label, or `<summary>`. I did not find a current conflict with info buttons, chart controls, accordions, or parent click handlers.
- Held-out modal controls are bound once behind `dialog._oss_bound`, so the persistent control listeners do not accumulate across dashboard rerenders.
- `encodeURIComponent(run.name)` is safe for the current published run names, and it is consistent with the existing viz-strip `runUrl()` pattern at `dashboard-public/index.html:2809`.

## Pass 3 review

### Pass 2 fixes verified

- **Held-out 404 loop mostly fixed for sustained traffic — `986a5a6` — `dashboard-public/index.html:5867`, `dashboard-public/index.html:5918`**: `_heldoutProbeStep()` now probes `model/sample-000.png` before starting the interval, so an old scored step with no published frame tree no longer immediately enters the 12 fps x 4 stream loop. If the probe succeeds but a later sample or non-model stream 404s, the per-image `error` handler calls `_heldoutShowMissingState()`, which clears `intervalHandle`, removes all image `src`s, sets `framesMissing`, and flips playback off at `dashboard-public/index.html:5831` through `dashboard-public/index.html:5844`. That can still emit one frame's worth of failed requests, but it does not sustain the storm.
- **Explicit close-button cleanup improved — `986a5a6` — `dashboard-public/index.html:6020`**: `closeHeldoutVideoModal()` now clears the playback interval, removes each retained image `src`, nulls `heldoutPlayer.imgs`, replaces the grid children, clears `run`/`steps`, resets indices, and clears `heldoutPlayer.framesMissing` at `dashboard-public/index.html:6036`. That fixes the main retained-bitmap/reference leak for the dismiss-button path.
- **PowerShell score-log parsing fixes verified — `c8011ce` / `985aebb` — `scripts/3080ti/heldout-eval-supervisor.ps1:50`, `scripts/3080ti/heldout-frames-backfill.ps1:91`**: both parsers now assign `ConvertFrom-Json` directly, reject `String` explicitly in the `IEnumerable` branch, and accept the PS 5.1 JSON-array case as an enumerable `Object[]`. The scalar fallback covers the PS 5.1 single-row unwrap to `PSCustomObject` via `.step`; malformed actual scalars add no steps, which is the right fail-closed behavior for these skip lists. A true `IDictionary`/hashtable would be enumerable and would miss the scalar fallback, but PS 5.1 `ConvertFrom-Json` does not return a hashtable for this path.

### New / remaining findings

- **HIGH — `986a5a6` — `dashboard-public/index.html:6045`**: native dialog close still bypasses the new full cleanup. Pressing `Esc` fires the dialog `close` listener, but that listener only removes `lightbox-open` and clears the interval; it does not call `closeHeldoutVideoModal()` or otherwise clear `heldoutPlayer.run`, `steps`, `imgs`, `framesMissing`, image `src`s, or the title text. Recommendation: factor the teardown into an idempotent helper and call it from both the dismiss button and the dialog `close` event, without recursively calling `dialog.close()`.
- **MEDIUM — `986a5a6` — `dashboard-public/index.html:6004`**: `_heldoutProbeStep(...).then(...)` is not session-guarded. If the modal is closed or reopened for another run/step before a probe resolves, the old probe callback can still render/schedule against the shared `heldoutPlayer`, or a stale failed probe can show the missing-frame state for the newer session. Recommendation: capture a generation token, run identity, and step before probing, then verify the token and `dialog.open` before mutating player state.
- **MEDIUM — `986a5a6` — `dashboard-public/index.html:5918`, `dashboard-public/index.html:6024`**: per-image `error` listeners are anonymous and not explicitly removed. The explicit close-button path likely releases them by removing `src`, replacing the grid, and nulling `heldoutPlayer.imgs`; however, any native-close path or stale in-flight image event can still call `_heldoutShowMissingState()` because the handler only checks global `heldoutPlayer.run && heldoutPlayer.steps.length`. Recommendation: tie handlers to the same session token used by the probe, or use assignable handler functions that are cleared during teardown.
- **LOW — `986a5a6` — `dashboard-public/index.html:5845`, `dashboard-public/index.html:5964`, `dashboard-public/index.html:6014`**: title-bar text is not cleared on close. Explicit close clears the heavy references, but the visible title can retain the last run name or the "frames not published yet" message until the next open. Recommendation: reset `heldout-video-title`, `heldout-video-source`, and slider labels in the shared teardown helper.

### What still looks good

- I did not find any other live PowerShell `ConvertFrom-Json` sites or `@(... ConvertFrom-Json ...)` collapse patterns outside `heldout-eval-supervisor.ps1` and `heldout-frames-backfill.ps1`. Other `@(...)` hits in the 3080 Ti scripts are literal CLI argument arrays or accumulator initialization.
- Dropping the comma-return in `985aebb` is correct for the supervisor caller: `@(StepsAlreadyEvaluated -scoreLog $scoreLog)` collects the function's pipeline-emitted ints into a flat array, so `-notcontains` compares actual step numbers instead of a single nested array object.
- The held-out missing-frame state is conservative once reached: `framesMissing` blocks `_heldoutAdvanceFrame()`, the play button cannot restart URL churn while the flag is set, and any mid-loop 404 stops further interval-driven requests.

## Pass 4 — deferred items addressed

- **HIGH score_log writer race fixed**: added `scripts/_score_log_io.py` and routed `scripts/sr_temporal_held_out.py`, `scripts/sr_train_temporal.py`, and `scripts/sr_train_gaussian_temporal.py` through it. The helper serializes access with `score_log.json.lock`, writes `score_log.json.tmp`, and commits with `os.replace`. Held-out appends still replace same-step rows and keep step order; legacy trainer flushes now merge under the lock and preserve existing same-step rows so stale in-memory trainer state cannot clobber newer held-out rows. Added `tests/test_score_log_io.py` with the requested 5-thread append race test.
- **MEDIUM run URL slug contract fixed**: `scripts/build_public_dashboard.py` now emits `run.slug`, rejects names that do not already match their deterministic storage slug, and reads staged run files from `runs/<slug>`. `scripts/watch_and_publish.sh` stages each source run into `runs/<slug>`, including GPU status writes. `dashboard-public/index.html` uses `run.slug || run.name` for viz and held-out frame URLs. The checked-in `dashboard-public/data.json` was minimally updated so all current run slugs equal their names.
- **LOW bare-strip slider bug fixed**: the compare modal now tracks a modal-level `bareStripFile`. Moving the global step slider updates bare-strip to the global nearest file, and moving an individual panel slider updates bare-strip to that panel's current file. Bare-strip updates immediately while enabled and uses the selected step when toggled on after slider movement.

Verification run:

- `venv-py312/bin/python -m pytest -q tests/test_score_log_io.py tests/test_public_dashboard_schema.py tests/test_repro_manifest.py` passed (`10 passed`).
- `venv-py312/bin/python -m pytest -q tests/test_score_log_io.py tests/sr/temporal/test_train_smoke.py tests/sr/gaussian_temporal/test_train_smoke.py` passed (`7 passed`).
- `venv-py312/bin/python scripts/build_public_dashboard.py --runs-dir dashboard-public/runs --out /tmp/oss-dashboard-current-build && venv-py312/bin/python tools/check_data_schema.py /tmp/oss-dashboard-current-build/data.json` passed; current generated slugs match names.
- `venv-py312/bin/python tools/check_data_schema.py dashboard-public/data.json`, `venv-py312/bin/python -m py_compile ...`, `bash -n scripts/watch_and_publish.sh`, and `git diff --check` passed.
- Playwright local dashboard check passed: after moving the global slider, bare-strip loaded `step-00000100.png`; after moving a per-panel slider, it switched to `step-00000500.png`.

Remaining gaps not addressed in this pass:

- The Pass 3 held-out video native-close/session-token findings remain open; this pass only covered the three deferred items requested above.
- Held-out frame PNG publication still uses the existing direct writer/copy behavior described in earlier findings; no `.complete` marker or atomic PNG publish protocol was added here.

## Pass 5 — final closure review

### Item-by-item verdict (PASS / FAIL / PARTIAL with rationale)

- **Score-log race: PARTIAL.** The new helper serializes cooperating callers correctly: `scripts/_score_log_io.py` uses a per-process thread lock plus `fcntl.flock`/`msvcrt.locking` on `score_log.json.lock`, releases the OS lock and closes the fd in `finally` blocks, reads/modifies/writes under that lock, and commits via `os.replace`. Read failures, JSON decode failures, write failures, and exceptions during sort/merge do not leak the held lock. A crashed process releases the kernel lock, so a leftover sidecar file is harmless; however, there is no timeout for a live wedged lock holder. The remaining blocker is compatibility with non-cooperating writers: `scripts/sr_v6_held_out.py` still does unlocked read/append/write against `score_log.json` and the same fixed `score_log.json.tmp`, so it can race with the new helper and lose rows or collide on temp files. Existing JSON-array score logs read normally; malformed non-list files still fail, and non-dict entries are silently dropped.
- **Slug rollout: PARTIAL.** Current published asset URLs use `run.slug || run.name`: viz strip/lightbox/compare paths go through `runUrl()`, and held-out frame URLs go through `runSlug()`. The watcher stages `SOURCE_DIR/<name>` into `runs/<slug>`, and all currently published slugs equal names, so existing `runs/<name>/*` R2 objects remain reachable. This is not a full migration path for path-significant names: `run_storage_slug()` computes a deterministic slug but rejects any name whose slug would differ, so future unsafe names fail with an actionable builder error rather than being mapped. The schema only requires `slug` to be a string; it does not validate non-empty/storage-safe/unique or relationship-to-name constraints. `dashboard-public/data.json` has `slug` for each of its six checked-in runs, but it is stale relative to the current seven-run source config and does not include `srcnn-v6.2-pico-002`.
- **Bare-strip sync: PASS for behavior, PARTIAL for coverage.** Actual resolution order is last writer wins: modal open seeds `bareStripFile` to the clicked/global initial file; moving panel A sets it to panel A's nearest file, moving panel B overwrites it, and toggling bare-strip on renders panel B's file. With bare-strip already enabled, moving the global slider updates all panels, sets `bareStripFile` from the global nearest file, and immediately calls `updateCompareBareStrip()`. Column dropdown and per-panel step-slider listeners remain intact; the pointer handler still exits on form controls. I found no automated test coverage for the bare-strip A-then-B-toggle order or global-slider live update.

### Any new regressions or concerns

- The score-log closure is not complete while `scripts/sr_v6_held_out.py` bypasses the sidecar lock and uses the same temp filename. This is a remaining multi-writer risk, not something I fixed in this pass.
- `tests/test_score_log_io.py` covers replacement, stale trainer merge, and a five-thread append race, but not multi-process `flock`/`msvcrt` behavior.
- `dashboard-public/data.json` validates today (`runs=6`) and contains slug fields, but its checked-in contents are older than the current dashboard source config that includes active `srcnn-v6.2-pico-002`.

Verification run:

- `python3 -m pytest -q tests/test_score_log_io.py tests/test_public_dashboard_schema.py tests/test_repro_manifest.py` passed (`8 passed, 2 skipped`).
- `python3 tools/check_data_schema.py dashboard-public/data.json` passed (`runs=6`).
- `git diff --check` passed.

### Sign-off

Not signed off as full Pass 4 closure. The bare-strip behavior fix is closed for the reviewed paths, and the slug change is safe for current identity slugs, but the score-log race remains partially open due to an unlocked direct writer. Per instruction, I stopped at documentation and did not make source changes.
