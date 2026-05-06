"""Tests for AA-2DGS object-space Mip filter."""

from __future__ import annotations

import math

import pytest
import torch

from oss.sr.v6.aa_2dgs_object_space_mip import object_space_mip_factor


def test_shape_correctness() -> None:
    scales = torch.ones(7, 2)
    out = object_space_mip_factor(scales, target_pixel_size=1.0, base_pixel_size=1.0)
    assert out.shape == (7,)


def test_shape_correctness_batched() -> None:
    scales = torch.ones(3, 5, 2)
    out = object_space_mip_factor(scales, target_pixel_size=1.0)
    assert out.shape == (3, 5)


def test_factor_in_unit_interval() -> None:
    scales = torch.tensor([[1.0, 1.0], [0.1, 5.0], [10.0, 0.5]])
    out = object_space_mip_factor(scales, target_pixel_size=1.0)
    assert (out > 0.0).all()
    assert (out <= 1.0).all()


def test_zoom_out_reduces_factor() -> None:
    """Larger target_pixel_size (zoom-out / minification) should NOT
    reduce the factor in this construction -- a wider integration box
    captures more mass. The AA-2DGS *opacity* attenuation comes from the
    erf saturating BELOW 1: when sigma >> w, the factor shrinks."""
    # Fix scales, vary target footprint relative to scale.
    scales = torch.tensor([[2.0, 2.0]])
    # Small w relative to sigma: integration box is small, capturing
    # only a fraction of the Gaussian mass.
    f_small = object_space_mip_factor(scales, target_pixel_size=0.5).item()
    # Large w relative to sigma: integration box captures most mass.
    f_large = object_space_mip_factor(scales, target_pixel_size=8.0).item()
    assert f_small < f_large


def test_factor_saturates_for_thin_gaussians() -> None:
    """When sigma << target_pixel_size, the splat is fully contained
    inside the pixel box and the factor saturates at 1."""
    scales = torch.tensor([[1.0e-3, 1.0e-3]])
    out = object_space_mip_factor(scales, target_pixel_size=10.0).item()
    assert out > 0.999


def test_factor_attenuates_for_large_sigma() -> None:
    """When sigma >> target_pixel_size, the per-axis erf argument is
    small and the factor approaches w / (sqrt(2*pi)*sigma) per axis."""
    scales = torch.tensor([[100.0, 100.0]])
    out = object_space_mip_factor(scales, target_pixel_size=1.0).item()
    assert out < 0.01


def test_closed_form_isotropic() -> None:
    """Hand-computed: scales=(1,1), w=1.
    erf(0.5/sqrt(2)) ~= 0.38292
    Factor = 0.38292^2 ~= 0.14663"""
    scales = torch.tensor([[1.0, 1.0]])
    out = object_space_mip_factor(scales, target_pixel_size=1.0).item()
    expected = math.erf(0.5 / math.sqrt(2.0)) ** 2
    assert abs(out - expected) < 1.0e-5


def test_closed_form_anisotropic() -> None:
    """scales=(1, 2), w=1 => erf(0.5/sqrt(2)) * erf(0.25/sqrt(2))."""
    scales = torch.tensor([[1.0, 2.0]])
    out = object_space_mip_factor(scales, target_pixel_size=1.0).item()
    expected = math.erf(0.5 / math.sqrt(2.0)) * math.erf(0.25 / math.sqrt(2.0))
    assert abs(out - expected) < 1.0e-5


def test_base_pixel_size_scaling() -> None:
    """Doubling base_pixel_size halves the effective w for the same
    target. So scales=(1,1), target=2, base=2 should match scales=(1,1),
    target=1, base=1."""
    scales = torch.tensor([[1.0, 1.0]])
    a = object_space_mip_factor(scales, target_pixel_size=2.0, base_pixel_size=2.0)
    b = object_space_mip_factor(scales, target_pixel_size=1.0, base_pixel_size=1.0)
    torch.testing.assert_close(a, b, atol=1.0e-6, rtol=1.0e-6)


def test_zero_sigma_floored() -> None:
    """Degenerate zero-sigma scale is floored; factor saturates at 1."""
    scales = torch.zeros(1, 2)
    out = object_space_mip_factor(scales, target_pixel_size=1.0).item()
    assert out > 0.999


def test_invalid_scales_shape_raises() -> None:
    with pytest.raises(ValueError):
        object_space_mip_factor(torch.zeros(5, 3), target_pixel_size=1.0)


def test_invalid_target_size_raises() -> None:
    with pytest.raises(ValueError):
        object_space_mip_factor(torch.ones(1, 2), target_pixel_size=0.0)


def test_invalid_base_size_raises() -> None:
    with pytest.raises(ValueError):
        object_space_mip_factor(torch.ones(1, 2), target_pixel_size=1.0, base_pixel_size=-1.0)


def test_bf16_safe() -> None:
    scales = torch.ones(3, 2, dtype=torch.bfloat16)
    out = object_space_mip_factor(scales, target_pixel_size=1.0)
    assert out.dtype == torch.bfloat16
    assert torch.isfinite(out).all()
