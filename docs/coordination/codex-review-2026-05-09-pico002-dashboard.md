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
