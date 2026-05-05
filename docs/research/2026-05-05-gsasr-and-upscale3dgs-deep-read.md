# 2026-05-05 — GSASR + upscale3dgs deep-read for v6 architecture

Author: research-assistant subagent
Method: WebFetch + GitHub REST API on the GSASR (commit `master` @ 2025-08-06) and upscale3dgs (commit `main` @ 2025) source trees. No local clone.

## TL;DR

GSASR is the closest published analogue to OSS v6's pixel-to-2D-Gaussian fusion: it predicts per-window Gaussian primitives from HAT-L features via a window-cross-attention decoder with explicit relative-position bias and a scale-conditioning token. The architecture is single-frame (no temporal canvas, no persistence), but its **WindowCrossAttn → GSSelfAttn → per-Gaussian MLP-head** stack is essentially the block diagram for v6's pixel↔Gaussian fusion stage and we should adopt it as a starting point rather than design from scratch. upscale3dgs is much narrower: a *non-learned* analytical replacement of finite-difference gradients in bicubic spline upsampling, applied on top of low-res 3DGS renders for 3-4× rendering speedups. It contributes essentially nothing to v6's neural architecture, but it does flag a free quality win for any cheap-render → upsample path and offers a useful contrast on what v6 actually buys versus a pure analytical gradient method.

## GSASR

### Architecture summary

(Sources: `TrainTestGSASR/basicsr/archs/fea2gs_arch.py`, `fea2gsropeamp_arch.py`, `hat_arch.py`, `hatrope_arch.py`; config `options/train/UltraPerformance/train_GSASR_HATL_amp_SA1B_bicubic_x1_16.yml`.)

Two-stage feed-forward model:

1. **Encoder (`network_g`):** `HATNOUP_ROPE_AMP` — HAT-L variant with the upsampler removed (`NOUP`), 12 Hybrid-Attention-Transformer blocks at `embed_dim=192`, `num_heads=6`, `window_size=16`, `mlp_ratio=2`, `compress_ratio=3`, `squeeze_factor=32`, `overlap_ratio=0.5`, ROPE mixed (`rope_mixed=True`, `rope_theta=10.0`), AMP-enabled. Produces a 64-channel feature map at LR resolution.
2. **Fea2GS decoder (`network_fea2gs`):** `Fea2GS_ROPE_AMP` (`fea2gsropeamp_arch.py`). Configured in the HAT-L recipe at `channel=192, num_heads=6, num_crossattn_blocks=4, num_crossattn_layers=4, num_selfattn_blocks=8, num_selfattn_layers=6, num_gs_seed=256, window_size=16, shuffle_scale1=2, shuffle_scale2=2`. Internals:
   - **Learnable Gaussian seeds**: `gs_embedding` and `pos_embedding` shaped `(num_gs_seed, channel)` shared across all windows.
   - **Scale conditioning**: scalar `1/scale` projected through a 2-layer MLP (`scale_mlp`) into a `channel`-dim token, broadcast as the K/V of an `nn.MultiheadAttention` cross-attention against the Gaussian queries (`gs_cross_attn_scale`). This is how arbitrary-scale (×1–×16) is injected.
   - **Window cross-attention (`WindowCrossAttn`)**: per-window query-side Gaussians attend to feature tokens within a `window_size × window_size` patch. The non-rope variant uses an explicit relative-position-bias table whose initialization gives extra weight (`+2.0`) to the closest source-target pixel pair (lines 84–92 of `fea2gs_arch.py`). The ROPE/AMP variant replaces this with mixed-frequency 2D rotary embeddings on Q/K (`init_random_2d_freqs`, `apply_rotary_emb_single`) and uses `F.scaled_dot_product_attention` (FlashAttention path). Shifted windows alternate `shift_size = window_size // 2` per layer (Swin-style).
   - **Gaussian self-attention (`GSSelfAttn`)**: shifted-window self-attention over the Gaussian set, again with relative-position bias.
   - **Density upsampling (`UPNet`)**: two `Conv2d → PixelShuffle(×2)` stages → 16× more Gaussians than `num_gs_seed` (so `num_gs_seed=256` per 16×16 LR window yields 4096 Gaussians per window).
   - **Five parallel MLP heads** decode each Gaussian into `(σx, σy)`, `ρ`, `α`, `(R,G,B)`, `(μx, μy)`. The means are added to a fixed reference grid (`get_N_reference_points`), so heads predict offsets, not absolute coords.
3. **Rasterizer**: custom CUDA scale-aware 2D splatter installed via `setup_gscuda.py`. Renders RGB by sampling continuous Gaussians at arbitrary HR target grid. Supports `tile_process` and `dmax` clipping (`dmax=0.5` in the HAT-L recipe).

Training: L1 only, Adam lr=2e-4, cosine-ish multistep at 250k/400k/450k/475k of 500k iters, EMA 0.999, batch size 8/GPU, 64×64 LR crops, scales sampled `[1.0, 16.0]`, AMP on, no LPIPS / GAN loss.

### What OSS should adopt

1. **Gaussian-seed + window-cross-attention pattern as the v6 fusion module.** The `WindowCrossAttnLayer.forward` is exactly v6's "cross-attention pixel↔Gaussian fusion" — learnable Gaussian queries cross-attending into HAT feature windows with relative-position bias to break translation symmetry. Hidden dim 192, 6 heads, 4 blocks × 4 layers is a known-good operating point with HAT-L on x1–x16 SR.
2. **Reference-grid + offset parameterization for Gaussian means.** GSASR predicts `μ_offset` and adds a fixed sub-pixel reference grid (`get_N_reference_points`). This is far more stable than predicting absolute means and matches what v6 needs for a *persistent canvas anchored to a screen-space lattice*.
3. **Five parallel small MLP heads** instead of one fat head. Decouples covariance / opacity / colour gradients during training.
4. **Scale-conditioning via a separate cross-attention** (`gs_cross_attn_scale`) rather than concat. v6 does not need arbitrary scale, but it does need *temporal-step* / *delta-time* conditioning, and this same pattern — scalar → MLP → MHA-K/V — is the right vehicle for it.
5. **ROPE-mixed + FlashAttention path** (`fea2gsropeamp_arch.py`). The relative-bias table version is a strict pedagogical baseline; for production v6 the ROPE/AMP variant is the right pick (FlashAttention drop-in, learnable per-head 2D frequencies).
6. **Density upsampling via PixelShuffle stack** to multiply Gaussian count without paying attention cost on the dense set. v6 should adopt this *before* its STVS pruning pass: predict at low density, upsample, then prune, rather than predicting dense from the start.
7. **HAT-L config for the encoder is directly copyable** (`embed_dim=192, depths=[6]*12, num_heads=[6]*12, window_size=16, mlp_ratio=2, ROPE`). The published HATL_SA1B weights are usable as initialization.

### What OSS should skip

1. **Per-frame independence.** GSASR re-predicts the entire Gaussian set every forward. v6's whole point is a persistent canvas — do not adopt their stateless inference loop.
2. **L1-only loss.** The HAT-L Ultra recipe uses pure L1; reported LPIPS (0.2381 on x4) and DISTS (0.1268) are mediocre because of it. v6 must keep its perceptual + temporal stability terms.
3. **`num_gs_seed=256` × `shuffle_scale=4` → ~4k Gaussians per 16×16 LR window** is wildly over-budget for streaming. v6 needs to operate at a fraction of this count and rely on STVS to prune; the GSASR density is feasible only because they tile-process offline.
4. **`tile_process: True` validation hack.** They process 64×64 tiles with 8-pixel overlap to fit memory. v6 cannot tile across temporal frames without breaking temporal consistency.
5. **The `dmax`/`step_size`/`mode='scale_modify'` machinery** in the rasterizer is specific to bicubic-downsample-aware arbitrary-scale rendering. Not directly relevant to v6's fixed-scale temporal upsample.
6. **The basicsr training harness.** Heavy, monolithic, not aligned with our research-loop pipeline. Borrow the model files, not the trainer.

### Numbers

(All from README; reported on standard SR benchmarks at ×4 unless noted. PSNR/SSIM on Y-channel.)

| variant | PSNR | SSIM | LPIPS | DISTS | dataset |
|---|---|---|---|---|---|
| EDSR enhanced | 31.04 | 0.8515 | 0.2512 | 0.1307 | DF2K |
| RDN enhanced | 31.10 | 0.8525 | 0.2482 | 0.1296 | DF2K |
| SwinIR enhanced | 31.17 | 0.8541 | 0.2456 | 0.1288 | DF2K |
| **HAT-L Ultra** | **31.31** | **0.8570** | **0.2381** | **0.1268** | SA1B |

Repo provides **no FPS, parameter count, or VRAM numbers** in the README. Paper presumably has them; we did not fetch the PDF body. Inference uses `--AMP_test` and `--tile_process` flags for memory.

### Action items for OSS v6

1. Pull `TrainTestGSASR/basicsr/archs/fea2gsropeamp_arch.py` into `oss/v6/models/fusion.py` as the starting point for the pixel↔Gaussian cross-attention block. Strip the basicsr `ARCH_REGISTRY` decorator. Keep the ROPE math, drop the relative-position-bias table.
2. Adopt the **5-head parallel MLP decoder** (`mlp_block_sigma / rho / alpha / rgb / mean`) and the **reference-grid + offset μ parameterization** verbatim. These are local design choices with no architectural risk.
3. Initialize v6's HAT-Base encoder from the **HATL_SA1B checkpoint** (HuggingFace, May 2025 release). Test whether HAT-L's 12×6 depths can be safely truncated to HAT-Base's 6×6 by dropping odd-indexed RHTBs.
4. Replace GSASR's `scale_mlp(1/scale)` with `temporal_mlp(Δt, view_motion_vec)` for v6's temporal conditioning, but keep the same MHA-as-conditioning pattern.
5. Steal `setup_gscuda.py` and the rasterizer kernel as a read-only reference for our own covariance-resampling rasterizer; do not vendor it (Apache-2.0 permits, but their kernel is bicubic-downsample-aware which we don't want).
6. Memo open question: GSASR uses **256 seeds × 16× density expansion = 4096 Gaussians per 16×16 LR window**. What is OSS's seed count budget per window after STVS pruning? Need a bench memo before fixing the v6 fusion module config.

## upscale3dgs

### Architecture summary

(Sources: `README.md`, project page `niedermayr.dev/upscale3dgs/`, arXiv 2503.14171.)

There is no neural network. The contribution is a **modified bicubic spline interpolation** for upsampling low-res 3DGS renders to higher resolution. Standard bicubic splines need pixel-value gradients at sample points; classical implementations estimate them with finite differences. The paper's substitution: since the 3DGS forward pass is differentiable and Gaussians have closed-form image-space gradients (∂I/∂x, ∂I/∂y of each Gaussian's contribution), they pass *exact analytical gradients* directly into the bicubic kernel during the upsample step. This sharpens edges versus finite-difference bicubic and is fully differentiable, enabling integration into 3DGS gradient-based optimization. The README is a one-pager; training code is "coming soon"; the released artifact is a Brush-based web viewer (`KeKsBoTer/brush`).

### What OSS should adopt

1. **The principle**: where a 2D Gaussian's image-space gradient is analytically available (and in our rasterizer it is), prefer it over numerical estimation in any post-process upsampling stage. If v6 ever does a "render at 1080p, upsample to 4K with bicubic" cheap path for low-confidence regions or for a debug visualization, this is a free quality bump.
2. **The framing is a clean ablation baseline.** A "v6 minus the neural fusion, plus analytical-gradient bicubic" branch is a good cheap baseline to publish against — it isolates what the cross-attention actually buys.

### What OSS should skip

1. **Everything else.** No neural model, no temporal mechanism (despite the ICCV abstract framing, the public page makes no temporal claims and we should not assume any), no covariance resampling, no training. The technique is at the wrong layer of v6.
2. **The Brush viewer**: a Rust/WGSL 3DGS viewer; useful as a reference codebase if we ever want WGSL splatting, but not in scope for v6.

### Numbers

- **3-4× rendering speedup** vs full-res 3DGS (render at low res, upsample analytically vs render at full res).
- **Up to 8× upscaling demonstrated** in their video comparisons.
- **No PSNR / LPIPS / SSIM tables** on the project page or README. The arXiv paper has them but we did not fetch the body. Compared qualitatively against bicubic and NinaSR1.
- **No reported temporal stability metric** despite our prior (the prompt's claim of "explicit temporally stable upscaling claim" is **not supported** by the README or project page we retrieved — flagging as a possible briefing inaccuracy. Would need PDF body).

### Action items for OSS v6

1. **Do not block on this repo.** It is a sharpening trick, not architecture.
2. If/when we add a "render-low, upsample-analytically" cheap path for thin-client or fallback rendering, fetch the arXiv PDF and reproduce the analytical-gradient bicubic kernel — it is ~50 lines of math + a CUDA kernel.
3. Verify the temporal-stability claim by reading the arXiv PDF body before citing this work as temporal prior art. Our prompt-side belief that the paper claims temporal stability appears wrong from public materials.

## Synthesis: what these two repos jointly tell us about v6

These two papers occupy opposite ends of a "how much neural machinery between Gaussians and the displayed pixel" axis, and v6 sits between them.

**GSASR** is heavy: HAT-L encoder + 4 cross-attn blocks × 4 layers + 8 self-attn blocks × 6 layers + dense MLP heads + custom CUDA rasterizer, all to predict ~4k image-conditioned 2D Gaussians per 16×16 LR window from a single LR frame. It pays for that with PSNR (31.31 on x4 HAT-L Ultra) and arbitrary-scale (×1–×16). It is stateless. Every frame is from scratch.

**upscale3dgs** is light: zero learned parameters, a closed-form analytical-gradient bicubic kernel applied as a post-process to a low-res 3DGS render. 3–4× speedup at modest quality cost.

The shared insight is that **2D Gaussians' analytical image-space gradients are the load-bearing mathematical object in both papers**. GSASR uses them implicitly through the differentiable splatter during training; upscale3dgs uses them explicitly to replace finite differences. v6's covariance-resampling step lives in the same algebra and we should ensure our rasterizer exposes ∂I/∂x and ∂I/∂y as a side output, not as a backward-pass-only quantity.

The contrast tells us where v6's *novel* contribution actually is. GSASR shows that pixel↔Gaussian cross-attention works as a feed-forward primitive. upscale3dgs shows that you can extract spatial sharpening from Gaussians with no learning at all. **Neither has a persistent canvas. Neither has a temporal score for pruning. Neither does cross-frame fusion of a Gaussian set.** The thing v6 proposes that is unique in this comparison is the *temporal persistence* of the canvas across frames combined with STVS-based covariance resampling — i.e. what GSASR does once per frame, v6 does once at sequence start and then *updates* with the same fusion machinery.

This is good news architecturally: the per-frame fusion block is mostly de-risked by GSASR. The v6 risk surface is concentrated in (a) the persistence/update logic, (b) STVS pruning, and (c) the cross-frame covariance resampling. We should not waste cycles re-deriving the cross-attention design.

It is also a useful sanity check on scope. GSASR's HAT-L Ultra is a 500k-iter SA1B-trained model that does single-frame x4 SR at 31.31 PSNR. v6's headline metric should not be single-frame PSNR — we will lose to GSASR there at any given parameter budget because we are spending capacity on temporal mechanics they do not have. v6's headline metric must be temporal stability or temporal-aware perceptual quality on a benchmark GSASR cannot run.

## References

- ChrisDud0257/GSASR, ICCV 2025. github.com/ChrisDud0257/GSASR. Default branch `master`, last push 2025-08-06. Apache-2.0. arXiv:2501.06838.
  - `TrainTestGSASR/basicsr/archs/fea2gs_arch.py` (25 KB) — non-rope cross-attn decoder.
  - `TrainTestGSASR/basicsr/archs/fea2gsropeamp_arch.py` (28 KB) — ROPE + AMP + FlashAttention variant; **production reference**.
  - `TrainTestGSASR/basicsr/archs/hatrope_arch.py` (43 KB) — HAT encoder with ROPE.
  - `TrainTestGSASR/options/train/UltraPerformance/train_GSASR_HATL_amp_SA1B_bicubic_x1_16.yml` — full HAT-L recipe.
  - `setup_gscuda.py` (root) — CUDA rasterizer build entry point.
- KeKsBoTer/upscale3dgs, ICCV 2025. github.com/KeKsBoTer/upscale3dgs. arXiv:2503.14171. Project page niedermayr.dev/upscale3dgs (redirects from keksboter.github.io). License not stated in README. Training code "coming soon" as of fetch (2026-05-05). Companion viewer github.com/KeKsBoTer/brush.

## Caveats on this memo

- All architectural detail is from public source files and README. Neither paper PDF body was fetched; the GSASR paper likely has FPS/param numbers in its tables that the README omits.
- The prompt's claim that upscale3dgs makes an "explicit temporally stable upscaling" claim is **not corroborated** by the public materials we read. Flag for verification.
- The prompt's claim that GSASR has an "online demo (June 2025)" is corroborated indirectly — the repo has `demo_gr.py` (Gradio) and HuggingFace-hosted weights — but we did not visit a live demo URL.
- We did not clone either repo. CUDA kernel internals (rasterizer math, scale-aware sampling logic) were not inspected — only Python wrappers and configs.
