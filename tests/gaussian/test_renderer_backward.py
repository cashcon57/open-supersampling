"""Sprint 1 / T1.5 — differentiable backward test.

Validates that the Rasterizer is end-to-end differentiable so it can be used
inside a training loop where the network predicts Gaussian parameters and we
backprop image-space loss through the renderer to the network.

Reference-backend tests run anywhere. CUDA tests are gated on gsplat being
importable AND a CUDA device being present.
"""

from __future__ import annotations

import pytest
import torch

from oss.gaussian.renderer import GaussianBatch, Rasterizer


def _make_batch(device: torch.device, requires_grad: bool = True) -> GaussianBatch:
    """Tiny batch with all parameters trainable. Gaussians use asymmetric
    scales so rotation gradients are non-zero (a circular Gaussian is
    rotation-invariant — its rendered output has zero derivative w.r.t. θ)."""
    xy = torch.tensor([[8.0, 8.0], [24.0, 8.0], [16.0, 24.0]],
                      device=device, requires_grad=requires_grad)
    scale = torch.tensor([[3.0, 1.0], [1.0, 3.0], [2.5, 1.5]],
                         device=device, requires_grad=requires_grad)
    rot = torch.tensor([0.3, 0.5, 1.0], device=device, requires_grad=requires_grad)
    feat = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                        device=device, requires_grad=requires_grad)
    return GaussianBatch(xy=xy, scale=scale, rot=rot, feat=feat)


def test_reference_backend_gradients_flow_to_all_params() -> None:
    """A scalar loss on the rendered output should produce non-zero grads on
    every Gaussian parameter tensor."""
    g = _make_batch(torch.device("cpu"))
    r = Rasterizer(force_backend="reference")
    out = r(g, output_hw=(32, 32))
    loss = out.sum()
    loss.backward()
    for name, t in (("xy", g.xy), ("scale", g.scale), ("rot", g.rot), ("feat", g.feat)):
        assert t.grad is not None, f"{name}.grad is None"
        # At least one element of the grad must be non-zero.
        assert torch.any(t.grad != 0), f"{name}.grad is all zeros — backward not flowing"


def test_reference_backend_optimization_moves_gaussian_toward_target() -> None:
    """Single Gaussian, target located off-center. After a few SGD steps, the
    Gaussian's position should move closer to the target."""
    target_xy = torch.tensor([24.0, 24.0])
    g = GaussianBatch(
        xy=torch.tensor([[8.0, 8.0]], requires_grad=True),
        scale=torch.tensor([[2.0, 2.0]], requires_grad=True),
        rot=torch.tensor([0.0], requires_grad=True),
        feat=torch.tensor([[1.0]], requires_grad=True),
    )
    # Build a target image with a Gaussian peaked at target_xy.
    target_g = GaussianBatch(
        xy=target_xy.unsqueeze(0),
        scale=torch.tensor([[2.0, 2.0]]),
        rot=torch.tensor([0.0]),
        feat=torch.tensor([[1.0]]),
    )
    r = Rasterizer(force_backend="reference")
    target_image = r(target_g, output_hw=(32, 32)).detach()

    initial_dist = torch.norm(g.xy[0] - target_xy).item()
    opt = torch.optim.Adam([g.xy, g.scale, g.rot, g.feat], lr=0.5)
    for _step in range(20):
        opt.zero_grad()
        out = r(g, output_hw=(32, 32))
        loss = torch.nn.functional.mse_loss(out, target_image)
        loss.backward()
        opt.step()

    final_dist = torch.norm(g.xy[0].detach() - target_xy).item()
    assert final_dist < initial_dist, (
        f"Gaussian did not move toward target. initial_dist={initial_dist:.2f} "
        f"final_dist={final_dist:.2f}"
    )


# --- CUDA backend backward test ---------------------------------------------

cuda_available = torch.cuda.is_available()
try:
    from gsplat import rasterize_gaussians_sum  # noqa: F401

    gsplat_available = True
except Exception:
    gsplat_available = False


@pytest.mark.gpu
@pytest.mark.skipif(not (cuda_available and gsplat_available),
                    reason="CUDA / gsplat not available")
def test_cuda_backend_gradients_flow() -> None:
    """CUDA backend must also propagate gradients to Gaussian parameters.

    Uses a larger image (128×128) so the normalized-coords convention
    inside the wrapper produces non-degenerate Gaussian footprints. At
    32×32 with the small fixture Gaussians, the projection step yields
    very few tile hits → near-zero gradient signal.
    """
    # Asymmetric scales so rotation gradient is non-zero (a circular
    # Gaussian is rotation-invariant).
    g = GaussianBatch(
        xy=torch.tensor([[40.0, 40.0], [88.0, 40.0], [64.0, 88.0]],
                        device="cuda", requires_grad=True),
        scale=torch.tensor([[12.0, 4.0], [4.0, 12.0], [10.0, 6.0]],
                           device="cuda", requires_grad=True),
        rot=torch.tensor([0.3, 0.5, 1.0], device="cuda", requires_grad=True),
        feat=torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                          device="cuda", requires_grad=True),
    )
    r = Rasterizer(force_backend="cuda")
    out = r(g, output_hw=(128, 128))
    out.sum().backward()
    for name, t in (("xy", g.xy), ("scale", g.scale), ("rot", g.rot), ("feat", g.feat)):
        assert t.grad is not None, f"{name}.grad is None"
        assert torch.any(t.grad != 0), f"{name}.grad is all zeros"


@pytest.mark.gpu
@pytest.mark.skipif(not (cuda_available and gsplat_available),
                    reason="CUDA / gsplat not available")
def test_cuda_backend_optimization_converges() -> None:
    """Same as the reference convergence test but on CUDA. Uses a 128×128
    image so the normalized-coords convention produces stable gradients."""
    target_xy = torch.tensor([96.0, 96.0], device="cuda")
    g = GaussianBatch(
        xy=torch.tensor([[32.0, 32.0]], device="cuda", requires_grad=True),
        scale=torch.tensor([[12.0, 12.0]], device="cuda", requires_grad=True),
        rot=torch.tensor([0.0], device="cuda", requires_grad=True),
        feat=torch.tensor([[1.0]], device="cuda", requires_grad=True),
    )
    target_g = GaussianBatch(
        xy=target_xy.unsqueeze(0),
        scale=torch.tensor([[12.0, 12.0]], device="cuda"),
        rot=torch.tensor([0.0], device="cuda"),
        feat=torch.tensor([[1.0]], device="cuda"),
    )
    r = Rasterizer(force_backend="cuda")
    target_image = r(target_g, output_hw=(128, 128)).detach()

    initial_dist = torch.norm(g.xy[0].detach() - target_xy).item()
    opt = torch.optim.Adam([g.xy, g.scale, g.rot, g.feat], lr=2.0)
    for _step in range(50):
        opt.zero_grad()
        out = r(g, output_hw=(128, 128))
        loss = torch.nn.functional.mse_loss(out, target_image)
        loss.backward()
        opt.step()
    final_dist = torch.norm(g.xy[0].detach() - target_xy).item()
    assert final_dist < initial_dist
