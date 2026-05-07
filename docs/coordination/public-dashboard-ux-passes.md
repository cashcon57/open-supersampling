# Public dashboard UX passes

Pass 1/5 - above-the-fold clarity. Screenshots showed the required status cards and measured number above the fold, but the pitch read as a DLSS/FSR critique rather than a one-sentence project description. The dashboard now uses a concise OpenSuperSampling pitch focused on unified super-resolution and frame extrapolation. After screenshots: `/tmp/pass-1-after-desktop.png`, `/tmp/pass-1-after-tablet.png`, `/tmp/pass-1-after-mobile.png`.

Pass 2/5 - information density vs whitespace. Desktop screenshots showed the at-a-glance cards stretching to the GPU panel height, leaving empty space that looked unfinished. The summary grid now aligns items to their natural height so the top section reads tighter. After screenshots: `/tmp/pass-2-after-desktop.png`, `/tmp/pass-2-after-tablet.png`, `/tmp/pass-2-after-mobile.png`.

Pass 3/5 - mobile experience. The 390x844 screenshot showed horizontal overflow from chart and viz containers inside the open run accordion. The run detail grid, chart wrapper, and nested cards now use `min-w-0` and canvas width guards; Playwright reported mobile `scrollWidth` equal to the viewport width. After screenshots: `/tmp/pass-3-after-desktop.png`, `/tmp/pass-3-after-tablet.png`, `/tmp/pass-3-after-mobile.png`.

Pass 4/5 - loss-curve legibility. The chart used separate axes, but the right axis was labeled only `LPIPS` even though it also carried Charbonnier, making the component series ambiguous. The right axis is now labeled `component loss`, with the component lines styled separately from total loss. After screenshots: `/tmp/pass-4-after-desktop.png`, `/tmp/pass-4-after-tablet.png`, `/tmp/pass-4-after-mobile.png`.

Pass 5/5 - accordion skim signal. Closed history rows showed run names and statuses, but did not clearly advertise what a reviewer would get by opening them. Historical summaries now call out the measured v5 numbers, v4 SRGD/latency baseline, and v6 grid-artifact evidence, with an `Open details` cue on desktop. After screenshots: `/tmp/pass-5-after-desktop.png`, `/tmp/pass-5-after-tablet.png`, `/tmp/pass-5-after-mobile.png`.
