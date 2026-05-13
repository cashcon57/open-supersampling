# v7 Pico student — ERM (Efficient Reshading Module) block spec

**Date:** 2026-05-13
**Status:** Phase 4 design memo. The Pico **student** model doesn't exist yet — pico-005 (the current training run) is the **teacher**. This memo documents the architectural target for the eventual ≤0.4M-param CNN student that ships to end users.
**Why now:** STSS shipped a concrete architecture (local 5×5 ReLU-linear attention as their cross-attention block) hitting **4.35 ms @ 1080p, 0.4M params, PSNR 35 / LPIPS 0.018** on RTX 3090 fp16. That's the latency / quality bar OSS Heavy student must match.

## What ERM is

STSS's "Efficient Reshading Module" is a cross-attention block that fuses LR features with HR canvas features. Three design choices make it cheap enough for real-time:

1. **Local 5×5 attention window.** Each output pixel attends only to a 5×5 neighborhood of the LR feature map, not the full LR plane. Reduces attention from O(N²) to O(N · k²) with k=5. For 1080p HR, that's ~2M ops/pixel instead of ~2B.
2. **ReLU-linear attention instead of softmax.** The attention scores go through `ReLU(qK^T)` not `softmax(qK^T / √d) V`. No exp(), no normalization. About 3× faster than softmax attention on the same FLOPs budget. Trades expressivity for throughput.
3. **Shared QKV projection across all heads.** STSS uses a single linear `nn.Conv2d(F, 3F, 1)` and splits, vs three separate linears. Saves both params and bandwidth.

The full STSS block is roughly:

```
def erm_block(lr_feat, hr_canvas_feat):
    # lr_feat:  (B, F, H_lr, W_lr)
    # hr_canvas_feat: (B, F, H_hr, W_hr) — pre-upsampled canvas render
    q = conv1x1(hr_canvas_feat)                                # (B, F, H_hr, W_hr)
    kv = conv1x1(lr_feat) -> upsample bilinear -> split (k, v) # 2 × (B, F, H_hr, W_hr)
    # 5x5 local attention: for each HR pixel, score against 5x5 LR-neighborhood-mapped K
    scores = relu(unfold_5x5(k) ⊙ q.unsqueeze(-1))             # (B, F, 25, H_hr, W_hr)
    weights = scores / (scores.sum(dim=2, keepdim=True) + 1e-6)
    out = (weights * unfold_5x5(v)).sum(dim=2)                 # (B, F, H_hr, W_hr)
    return out + hr_canvas_feat   # residual
```

## How it maps to OSS

The Pico student must:

- Take the same **9-channel LR input** (RGB + depth + motion + normals; matches teacher's input format)
- Take an **HR canvas feature map** produced by some compact persistent state (the OSS canvas — but for the student we're allowed to use a quantized / pruned version of the teacher's canvas)
- Produce **HR RGB** output
- Run in <2 ms at 1080p HR on a target consumer GPU

The OSS student doesn't need to re-derive the time-slice math — at inference, we sample the canvas at the current frame's t, get an HR feature map, and pass that to the student. The student's job is "fuse LR + HR-canvas-features into HR RGB" — exactly what ERM does in STSS.

## Proposed OSS-ERM block (concrete)

```python
class OssERMBlock(nn.Module):
    """Local 5x5 ReLU-linear cross-attention block for the Pico student.

    Fuses LR backbone features with the HR-upsampled canvas feature map.
    Designed for <2ms 1080p inference on RTX 3060+.

    Args:
        feat_dim: Channel count (same on Q and KV side).
        window:   Spatial window size (5 = STSS default).
        heads:    Multi-head split. 4 heads at 64-channel feat_dim = 16 ch/head.
    """
    def __init__(self, feat_dim: int = 64, window: int = 5, heads: int = 4):
        super().__init__()
        assert feat_dim % heads == 0, "feat_dim must divide by heads"
        self.feat_dim = feat_dim
        self.window = window
        self.heads = heads

        # Shared QKV projection — STSS-style.
        self.qkv = nn.Conv2d(feat_dim, feat_dim * 3, kernel_size=1)
        self.proj = nn.Conv2d(feat_dim, feat_dim, kernel_size=1)
        # Bilinear LR -> HR upsample on KV before attention (same scale as backbone).
        self.scale = 2

    def forward(self, hr_feat: torch.Tensor, lr_feat: torch.Tensor) -> torch.Tensor:
        # hr_feat: (B, F, H_hr, W_hr) — query (from canvas render)
        # lr_feat: (B, F, H_lr, W_lr) — key + value source
        B, F_, H, W = hr_feat.shape

        # 1) QKV projection. Q from HR, K+V from LR upsampled.
        qkv_hr = self.qkv(hr_feat)
        q = qkv_hr[:, :F_]
        lr_up = F.interpolate(lr_feat, size=(H, W), mode='bilinear', align_corners=False)
        qkv_lr = self.qkv(lr_up)
        k = qkv_lr[:, F_:2*F_]
        v = qkv_lr[:, 2*F_:]

        # 2) Local 5x5 unfold on K and V around each HR pixel.
        pad = self.window // 2
        k_unf = F.unfold(k, kernel_size=self.window, padding=pad)  # (B, F*25, H*W)
        v_unf = F.unfold(v, kernel_size=self.window, padding=pad)
        k_unf = k_unf.view(B, F_, self.window**2, H, W)
        v_unf = v_unf.view(B, F_, self.window**2, H, W)

        # 3) ReLU-linear attention: relu(q · k) / norm. No softmax, no exp.
        scores = F.relu((q.unsqueeze(2) * k_unf).sum(dim=1, keepdim=True))  # (B, 1, 25, H, W)
        weights = scores / (scores.sum(dim=2, keepdim=True) + 1e-6)
        out = (weights * v_unf).sum(dim=2)                                  # (B, F, H, W)

        # 4) Output projection + residual.
        return hr_feat + self.proj(out)
```

Approx param count: 3 × (F × F × 1×1) for QKV + (F × F × 1×1) for proj = 4F². At F=64 that's **16,384 params per block**. Stack 4 of them = 65K. The rest of the ~0.4M budget goes to the stem CNN that produces `lr_feat` and the head that turns the fused HR feature map into RGB.

## How this fits the v7 Pico distillation plan

The Pico student is distilled from the v7 teacher (HAT-Tiny backbone + N-D canvas + parent-child + Mip-Splatting filters + everything in `oss/sr/v7/`). At training time:

1. Run the **teacher** forward on a (LR, HR_GT) pair → produces a teacher feature map at the composite-head input, plus the teacher's canvas at t=N
2. Run the **student** forward on the same LR + the teacher's canvas render → produces student RGB
3. **Distillation loss:**
   - L1(student_rgb, teacher_rgb)  ← primary supervision target
   - L1(student_feat, teacher_feat) at the composite-head input  ← feature distillation
   - L1(student_rgb, gt_hr) × 0.2  ← anchor to ground truth
   - LPIPS(student_rgb, gt_hr) × 0.5
   - Sobel HF on (student_rgb, gt_hr) × 0.1
   - RRM 2× weighting in synthetic disocclusion regions
4. The student does NOT learn its own spawner — it consumes the teacher's canvas at inference. The teacher is run sparsely (every N frames) to refresh the canvas; the student handles in-between frames.

This is the **MOAT-style architecture** the synthesis described: heavy teacher for canvas, light student for per-frame fusion.

## Inference-time wiring (target)

```
Every K frames (e.g. K=8, on a 60fps target):
    teacher: LR_t -> spawn -> canvas updated with new t-coord Gaussians

Every frame:
    canvas.render_at(t_now) -> HR canvas features (F=64, H_hr, W_hr)
    stem(LR_t)              -> LR features (F=64, H_lr, W_lr)
    for _ in num_erm_blocks:
        hr_feat = OssERMBlock(hr_feat, lr_feat)
    output_rgb = head(hr_feat) + bicubic(LR_t.rgb)
```

The teacher runs ~7.5 fps on a slow target host, the student runs 60+ fps. Canvas time-extrapolation via the V_xt cross-correlation in our Cholesky covariance smooths over the gaps between teacher updates — that's the value v7's native-time-axis primitive provides over a 2D-Gaussian alternative.

## Open questions

1. **Window size: 5 vs 7 vs 9.** STSS picks 5. With OSS canvas as a strong prior, we may be able to drop to 3 (one Gaussian's footprint of context, max). Need an ablation once the student trains.
2. **Heads: shared QKV vs per-head Q.** STSS shares; some recent papers report a quality bump from per-head Q at the same parameter budget. Cheap to ablate.
3. **Should the student bypass the canvas entirely for low-loss regions?** ClassSR-style adaptive routing — easy patches go through a fast bilinear path, hard patches through the ERM stack. Phase 4.b investigation.
4. **Quantization-aware training.** STSS reports fp16. We want fp8 / int8 for the embedded targets (mobile, Switch 2-class hardware). Stand-alone post-training quantization should work but needs validation.
5. **Distillation loss balance.** Heavy reliance on L1 between student/teacher features risks the student inheriting teacher artifacts. Need to ablate `L1(student_rgb, gt_hr)` vs `L1(student_rgb, teacher_rgb)` weight ratio.

## Status

This is a design memo only. No code lands until pico-005 (the teacher) produces a usable checkpoint. Phase 4 timeline:

- **t = pico-005 + ~6 days**: First teacher checkpoint at step 100K
- **t = + 1 week**: Implement OssERMBlock + student head + distillation trainer
- **t = + 2-3 weeks**: First student training run (`pico-005-student`)
- **t = + 4-5 weeks**: ONNX export + cross-vendor runtime validation (TensorRT FP8, DirectML, CoreML)

Tracking ID: `v7-pico-005-student` for the distillation run. Lives alongside `srcnn-v7.0-pico-005` in checkpoint dir.
