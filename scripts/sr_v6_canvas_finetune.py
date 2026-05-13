"""v6.3-fine: fine-tune the v6.2 composite_head with a learnable canvas-
fusion scalar + canvas-aware auxiliary loss. HAT-Tiny backbone is frozen.
GAN is disabled. Pure regression + perceptual + aux. Validates that the
v6.3 design ingredients (magnitude scaling + aux loss) shift training
behavior before we commit to v7.

Spec: docs/architecture/2026-05-12-v63-fine-finetune-spec.md

Usage:
    python scripts/sr_v6_canvas_finetune.py \\
        --ckpt-init E:\\checkpoints\\srcnn-v6.2-pico-002\\step-00100000.pt \\
        --tartanair-root E:\\datasets\\tartanair_extracted \\
        --output-dir E:\\checkpoints\\srcnn-v6.3-fine \\
        --steps 10000 --lr 1e-4 --canvas-scale-init 50 --lambda-aux 0.1
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn.functional as F


def _device(arg: str) -> str:
    if arg == "cuda" and not torch.cuda.is_available():
        print("CUDA unavailable, falling back to CPU.")
        return "cpu"
    return arg


def _charbonnier(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return torch.sqrt((pred - target) ** 2 + eps * eps).mean()


def _psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = F.mse_loss(pred.clamp(0, 1), target.clamp(0, 1)).item()
    if mse <= 0.0:
        return 99.0
    import math
    return float(-10.0 * math.log10(mse))


class CanvasFusionScaleWrapper(torch.nn.Module):
    """Holds a learnable scalar that multiplies canvas_hr at the
    composite_head input. The scalar is registered as a model parameter
    so the optimizer can update it.
    """
    def __init__(self, init_value: float):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(float(init_value)))


def install_aux_hook(model, fusion_scale: CanvasFusionScaleWrapper, aux_capture: dict):
    """Forward-pre-hook on composite_head that:
       - replaces canvas_hr with scale * canvas_hr (for the MAIN forward
         pass — affects ``out_main``)
       - captures BOTH the original canvas_hr (for aux pass) AND the
         scaled version (for main loss)

    The aux forward pass is run separately by the training loop using
    aux_capture['canvas_hr_scaled'].
    """
    head = model.composite_head
    feat_dim = int(model.feat_dim)
    canvas_dim = int(model.rasterizer.feature_dim)

    def hook(_module, inputs):
        x = inputs[0]
        if x.shape[1] != feat_dim + canvas_dim:
            return None
        refined_hr = x[:, :feat_dim]
        canvas_hr = x[:, feat_dim:]
        canvas_scaled = canvas_hr * fusion_scale.scale
        aux_capture["canvas_hr_scaled"] = canvas_scaled
        aux_capture["refined_hr"] = refined_hr
        return (torch.cat([refined_hr, canvas_scaled], dim=1),)

    return head.register_forward_pre_hook(hook)


def compute_aux_delta(model, aux_capture: dict):
    """Compute the aux-loss prediction by passing
    [zeros_like(refined_hr), canvas_hr_scaled] through composite_head.
    This is the "canvas-only" output that the aux loss penalizes when
    it can't match GT.
    """
    refined = aux_capture["refined_hr"]
    canvas_scaled = aux_capture["canvas_hr_scaled"]
    zero_refined = torch.zeros_like(refined)
    aux_input = torch.cat([zero_refined, canvas_scaled], dim=1)
    return model.composite_head(aux_input)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt-init", required=True, type=Path)
    parser.add_argument("--tartanair-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--canvas-scale-init", type=float, default=50.0)
    parser.add_argument("--lambda-aux", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--ckpt-every", type=int, default=1000)
    args = parser.parse_args()

    device = _device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    from scripts.sr_temporal_held_out import _load_temporal
    from oss.gaussian.data import TartanAirGaussianDataset
    from oss.sr.temporal.dataset import adapt_tartanair

    print(f"[finetune] loading init ckpt {args.ckpt_init.name}")
    model = _load_temporal(args.ckpt_init, device)
    model.train(True)

    # Freeze backbone parameters; only composite_head + fusion scalar trainable.
    backbone_params = 0
    trainable_params = 0
    for name, p in model.named_parameters():
        if "composite_head" in name:
            p.requires_grad_(True)
            trainable_params += p.numel()
        else:
            p.requires_grad_(False)
            backbone_params += p.numel()
    print(f"[finetune] frozen backbone params: {backbone_params:,}")
    print(f"[finetune] trainable composite_head params: {trainable_params:,}")

    fusion_scale = CanvasFusionScaleWrapper(args.canvas_scale_init).to(device)
    trainable_params += 1  # the scalar
    print(f"[finetune] +1 trainable scalar (canvas_fusion_scale)")

    aux_capture: dict = {}
    handle = install_aux_hook(model, fusion_scale, aux_capture)

    # Optimizer over composite_head params + the scalar.
    head_params = [p for p in model.composite_head.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(
        head_params + [fusion_scale.scale],
        lr=args.lr,
        weight_decay=1e-4,
    )

    # Dataloader.
    ds = TartanAirGaussianDataset(root=args.tartanair_root, scale=2.0)
    print(f"[finetune] TartanAir dataset size: {len(ds)}")
    g = torch.Generator().manual_seed(42)

    def sample_pair_batch():
        from oss.sr.temporal import default_collate_pair
        items = []
        # Random pair: pick K random indices, build a (t, t+1) pair from
        # adjacent dataset items in the same trajectory.
        n = len(ds)
        for _ in range(args.batch_size):
            # Use simple random idx; the dataset's adjacency may not be
            # strictly trajectory-clean but TartanAir's _items list IS
            # ordered by trajectory.
            idx = int(torch.randint(low=0, high=n - 1, size=(1,), generator=g).item())
            t_sample = ds[idx]
            tp1_sample = ds[idx + 1]
            items.append({
                "t": t_sample,
                "tp1": tp1_sample,
            })
        return items

    history = []
    t0 = time.perf_counter()
    for step in range(1, args.steps + 1):
        batch_items = sample_pair_batch()
        model.train(True)
        optim.zero_grad()

        total_loss = 0.0
        n_in_batch = 0
        for it in batch_items:
            t_s = it["t"]
            p_s = it["tp1"]
            # Random spatial crop at args.patch_size HR
            hr_h, hr_w = t_s.gt_hr_frame.shape[-2:]
            ps = args.patch_size
            if hr_h < ps or hr_w < ps:
                continue
            ch = int(torch.randint(0, hr_h - ps + 1, size=(1,), generator=g).item())
            cw = int(torch.randint(0, hr_w - ps + 1, size=(1,), generator=g).item())

            def _crop_hr(x):
                return x[..., ch:ch + ps, cw:cw + ps]

            def _crop_lr(x):
                lr_ch = ch // 2
                lr_cw = cw // 2
                lr_ps = ps // 2
                return x[..., lr_ch:lr_ch + lr_ps, lr_cw:lr_cw + lr_ps]

            # Build 9-channel inputs (drop canvas hint; v6 uses 9).
            t_lr = _crop_lr(t_s.lr_frame).unsqueeze(0).to(device)
            t_depth = _crop_lr(t_s.depth).unsqueeze(0).to(device)
            t_motion = _crop_lr(t_s.motion).unsqueeze(0).to(device)
            t_normals = _crop_lr(t_s.normals).unsqueeze(0).to(device)
            t_lr_in = torch.cat([t_lr, t_depth, t_motion, t_normals], dim=1)

            p_lr = _crop_lr(p_s.lr_frame).unsqueeze(0).to(device)
            p_depth = _crop_lr(p_s.depth).unsqueeze(0).to(device)
            p_motion = _crop_lr(p_s.motion).unsqueeze(0).to(device)
            p_normals = _crop_lr(p_s.normals).unsqueeze(0).to(device)
            p_lr_in = torch.cat([p_lr, p_depth, p_motion, p_normals], dim=1)

            p_gt = _crop_hr(p_s.gt_hr_frame).unsqueeze(0).to(device)

            depth_hr_t = F.interpolate(t_depth, scale_factor=2.0, mode="bilinear", align_corners=False)
            depth_hr_tp1 = F.interpolate(p_depth, scale_factor=2.0, mode="bilinear", align_corners=False)

            model.reset_state(torch.device(device))
            _ = model(lr_inputs=t_lr_in, motion_lr=None,
                      depth_hr_curr=depth_hr_t, depth_hr_prev=depth_hr_t,
                      frame_index=0)
            out_main = model(lr_inputs=p_lr_in, motion_lr=t_motion,
                             depth_hr_curr=depth_hr_tp1, depth_hr_prev=depth_hr_t,
                             frame_index=1).clamp(0, 1)

            # Aux: run composite_head with refined_hr=0
            delta_aux = compute_aux_delta(model, aux_capture)
            bicubic_hr = F.interpolate(p_lr, scale_factor=2.0, mode="bicubic", antialias=True, align_corners=False).clamp(min=0.0)
            out_aux = (bicubic_hr + delta_aux).clamp(0, 1)

            loss_char = _charbonnier(out_main, p_gt)
            loss_aux = _charbonnier(out_aux, p_gt)
            loss = loss_char + args.lambda_aux * loss_aux
            total_loss = total_loss + loss
            n_in_batch += 1

        if n_in_batch == 0:
            continue
        total_loss = total_loss / float(n_in_batch)
        total_loss.backward()
        optim.step()

        if step % args.log_every == 0 or step == 1:
            scale_val = float(fusion_scale.scale.detach().cpu().item())
            psnr_val = _psnr(out_main[0].detach(), p_gt[0].detach())
            print(
                f"[step {step:5d}] loss={total_loss.item():.4f} "
                f"char={loss_char.item():.4f} aux={loss_aux.item():.4f} "
                f"scale={scale_val:.3f} psnr={psnr_val:.2f} "
                f"elapsed={time.perf_counter() - t0:.0f}s"
            )
            history.append({
                "step": step,
                "loss": float(total_loss.item()),
                "loss_char": float(loss_char.item()),
                "loss_aux": float(loss_aux.item()),
                "canvas_scale": scale_val,
                "psnr_train_crop": psnr_val,
            })

        if step % args.ckpt_every == 0 or step == args.steps:
            ckpt_path = args.output_dir / f"step-{step:08d}.pt"
            torch.save({
                "step": step,
                "composite_head_state": model.composite_head.state_dict(),
                "fusion_scale": fusion_scale.scale.detach().cpu().item(),
                "args": vars(args),
            }, ckpt_path)
            print(f"[finetune] ckpt -> {ckpt_path}")

    handle.remove()
    (args.output_dir / "history.json").write_text(json.dumps(history, indent=2))
    print(f"[finetune] done -> {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
