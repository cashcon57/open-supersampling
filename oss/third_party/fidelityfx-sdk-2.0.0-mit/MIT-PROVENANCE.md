# FidelityFX SDK 2.0.0 — MIT Provenance Record

**Vendored:** 2026-05-08
**Source repo:** `GPUOpen-LibrariesAndSDKs/FidelityFX-SDK` (official AMD GitHub)
**Source commit:** `01446e6a74888bf349652fcf2cbf5f642d30c2bf`
**Commit date:** 2025-08-18
**Commit message:** "AMD FidelityFX SDK 2.0.0"
**License at this commit:** standard MIT (see `docs/license.md` in this directory; verbatim text)
**Reproducibility tarball URL:** `https://api.github.com/repos/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK/tarball/01446e6a74888bf349652fcf2cbf5f642d30c2bf`

## Why this exact commit (and not the v2.0.0 release tag)

AMD's `v2.0.0` git tag currently points to commit `f4c1da8e7d...` dated **2025-08-20**. This is **not** the original SDK 2.0.0 commit; it is a follow-on cleanup commit that **scrubbed FSR 4 source** before the public release announcement.

The original SDK 2.0.0 commit is `01446e6a74...` dated **2025-08-18**, and it contains:

- `Kits/FidelityFX/upscalers/fsr4/` (183 files of FSR 4 HLSL operator source, including ml2code_runtime, FasterNet, fused conv kernels, etc.)
- `docs/license.md` containing standard MIT permission text granting use, copy, modify, merge, publish, distribute, sublicense, and sell rights

The scrubbed `f4c1da8e` commit removed `Kits/FidelityFX/upscalers/fsr4/` entirely. AMD then reset the `main` branch and `v2.0.0` tag to point at the scrubbed commit.

## How we know AMD force-pushed (forensic record)

Both commits exist in AMD's official repo. Both share the same parent (`c6efa6bf` = SDK 1.1.4). They are git siblings, not a linear history.

**Reproducibility — anyone can verify:**

```bash
# Original SDK 2.0.0 (with FSR 4 source, MIT)
gh api repos/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK/commits/01446e6a74888bf349652fcf2cbf5f642d30c2bf
# → returns valid commit, parent = c6efa6bf, message "AMD FidelityFX SDK 2.0.0"

# Scrubbed SDK 2.0.0 (current v2.0.0 tag)
gh api repos/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK/commits/f4c1da8e
# → returns valid commit, parent = c6efa6bf, message "AMD FidelityFX SDK 2.0.0"

# Both are reachable from the official repo via SHA. The scrubbed one is reachable
# via tag + branch; the original is reachable only by SHA (orphaned).

# Verify FSR 4 source presence at the orphan commit
gh api 'repos/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK/git/trees/01446e6a7488?recursive=1' \
  | jq -r '.tree[] | select(.path | test("upscalers/fsr4")) | .path' | wc -l
# → 217 files

# Verify FSR 4 source ABSENCE at the scrubbed commit
gh api 'repos/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK/git/trees/f4c1da8e?recursive=1' \
  | jq -r '.tree[] | select(.path | test("upscalers/fsr4")) | .path'
# → empty (only fsr4-named image files in docs/, no source)

# Verify MIT license at the orphan commit
gh api 'repos/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK/contents/docs/license.md?ref=01446e6a7488' \
  | jq -r '.content' | base64 -d | head
# → "...Permission is hereby granted, free of charge, to any person obtaining a copy
#       of this software and associated documentation files(the 'Software'), to deal
#       in the Software without restriction, including without limitation the rights
#       to use, copy, modify, merge, publish, distribute, sublicense, and/or sell..."
```

## Why MIT applies despite AMD's force-push

The MIT license is **irrevocable for already-distributed copies**. Standard MIT terms grant rights "to any person obtaining a copy" — once a copy is obtained under MIT, the recipient retains use/copy/modify/merge/publish/distribute/sublicense/sell rights in perpetuity. A subsequent license change by the original author does not retroactively revoke rights for prior recipients.

This is settled MIT-license interpretation — it is the same principle that allows old MIT-licensed software (e.g., XFree86, BSD networking utilities) to remain freely usable indefinitely even after upstream relicensing or deprecation.

AMD's force-push:
- Does not delete the orphan commit (it remains in AMD's repository, reachable via direct SHA query)
- Does not retroactively unmake the MIT grant on the orphan commit
- Does not affect copies that were obtained between 2025-08-18 (push) and ~2025-08-20 (force-push)
- Does affect what new recipients receive *via the v2.0.0 tag or main branch* (those now point at the scrubbed commit, which has a different — though still MIT — license file)

This OSS project obtained its copy directly from AMD's GitHub repo on 2026-05-08 by querying the orphan commit by SHA. The license at that commit is MIT. Per standard MIT terms, OSS retains MIT rights to this snapshot in perpetuity.

## Subsequent license changes (informational)

After the force-push, AMD continued to use MIT in `docs/license.md` for SDK 2.1.0 (committed 2025-12-09 at SHA `0836aa7f`). At some point between SDK 2.1.0 and the current main HEAD, AMD changed `docs/license.md` to a substantially more restrictive license text:

> "...Permission... to install, reproduce, copy and distribute copies of the Software, **in binary form only**... **No reverse engineering, decompilation, or disassembly of this Software is permitted.**"

This restrictive license governs only what AMD distributes *under that license file* — it does not retroactively re-license prior MIT-licensed snapshots. Anyone who obtained a copy under any MIT-licensed commit (including this one) retains MIT rights to that snapshot.

## Selective vendoring rationale

The full SDK 2.0.0 tarball is ~129 MB. To keep OSS repo size manageable, this vendored copy includes only:

- `docs/license.md` (REQUIRED for MIT compliance — see License Preservation section below)
- `readme.md` (upstream readme; for context)
- `docs/techniques/super-resolution-{ml,temporal,upscaler}.md` (FSR 1/2/3/4 documentation)
- `Kits/FidelityFX/upscalers/fsr4/` (FSR 4 HLSL operator source — the primary vendored asset)
- `Kits/FidelityFX/upscalers/fsr3/` (FSR 3 source for cross-reference and reproducible comparison)
- `Kits/FidelityFX/signedbin/` (FSR 4 + FSR 3 frame-gen + loader DLLs, MINUS .pdb debug symbols)
- `Kits/FidelityFX/api/` (the FidelityFX API headers needed to integrate)

Excluded for size (all available from the upstream tarball if needed later):
- `.pdb` debug symbols (~11 MB)
- `Kits/Cauldron2/` (the framework — only needed if we build the sample app for benchmarking; can be added later)
- `Samples/` (sample apps; same as above)
- `docs/techniques/media/` (large reference images, ~37 MB; the .md docs themselves are kept)
- Tools, build scripts, .gitlab-ci configs

This is consistent with MIT requirements: MIT requires preservation of the copyright + license notice; it does not require redistributing the full original work.

## License preservation requirements

Per MIT terms:

> "The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software."

The license file at `docs/license.md` (in this directory) is the verbatim MIT permission notice from AMD's `01446e6a` commit, including the AMD copyright header. **It MUST remain in this directory unmodified.** The OSS root `NOTICE` file references this directory; that reference must remain intact.

Any further redistribution of this vendored copy (including OSS releases that bundle this directory) must include the `docs/license.md` file or its content with attribution to AMD.

## Acceptable uses under this MIT grant

OSS may, with respect to the contents of this directory:

- Use, study, copy, port, modify, distribute the FSR 4 HLSL source and other SDK source files
- Sublicense modifications under any license compatible with MIT (Apache 2.0, MIT, GPL — note that downstream license choice affects only modifications and combinations, not the original)
- Distribute the FSR 4 binary DLLs (they were distributed within the MIT-licensed SDK)
- Use the FSR 4 binary as a benchmark target
- Use FSR 4 (binary or source) as a distillation teacher for OSS student model training
- Use insights from the FSR 4 architecture (FasterNet block, fused-op patterns, SqrSwish, FP16 NHWC + INT8 weights) in OSS's own architecture

OSS may NOT:

- Reverse-engineer the binary DLLs (still prohibited by AMD's binary EULA outside the SDK source release; though MIT permits source-side modifications, RE of the precompiled binaries is governed separately)
- Remove the AMD copyright notice from `docs/license.md` or otherwise dilute the MIT preservation requirement

## Followups (not blocking)

- **Archival mirror:** consider cloning AMD's repo to a private OSS-controlled mirror (cashcon57 GitHub org) so even if AMD eventually does prune the orphan commit (e.g., via aggressive `git gc --prune=now` on their server), OSS retains a verifiable upstream copy. As of 2026-05-08, AMD has NOT pruned the orphan commit; it is still reachable via direct SHA query.
- **Vendored tarball checksum:** record the tarball SHA-256 here for additional integrity verification independent of git SHA.
