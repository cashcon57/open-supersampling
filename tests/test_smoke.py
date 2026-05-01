"""Full-pipeline smoke test on tiny inputs. Runs in seconds on CPU."""
import torch
import torch.nn.functional as F

from oss.handoff import validate_handoff
from oss.model import OSSRG, OSS
from oss.model.adapter import PairedOSS
from oss.train.losses import CompositeLoss


def test_full_pipeline_smoke():
    H, W = 16, 16
    ossrg_model = OSSRG(tier="lite").train(False)
    oss_model = OSS(input_mode="features", scale_factor=2.0, tier="lite").train(False)
    pair = PairedOSS(ossrg_model, oss_model)
    loss_fn = CompositeLoss(w_lpips=0.0)  # skip LPIPS on CPU smoke

    noisy = torch.randn(1, 3, H, W)
    aux = torch.randn(1, 11, H, W)
    history = torch.randn(1, 3, H, W)
    depth = torch.randn(1, 1, H, W)
    motion = torch.randn(1, 2, H, W)

    rgb_lo, rgb_hi = pair(noisy=noisy, aux=aux, history=history, depth=depth, motion=motion)
    assert rgb_lo.shape == (1, 3, H, W)
    assert rgb_hi.shape == (1, 3, H * 2, W * 2)

    # Handoff tensor itself must validate.
    _, feats = ossrg_model(noisy, aux, history)
    validate_handoff(feats)

    target_hi = torch.randn(1, 3, H * 2, W * 2)
    target_lo = F.interpolate(target_hi, scale_factor=0.5, mode="bilinear", align_corners=False)
    loss = loss_fn(rgb_lo, target_lo) + loss_fn(rgb_hi, target_hi)
    loss.backward()
    assert loss.item() > 0
