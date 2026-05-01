import torch
from oss.valuation.metrics import psnr, ssim, lpips_dist


def test_psnr_identical():
    a = torch.rand(1, 3, 64, 64)
    p = psnr(a, a)
    assert p.item() > 50.0


def test_ssim_identical():
    a = torch.rand(1, 3, 64, 64)
    s = ssim(a, a)
    assert abs(s.item() - 1.0) < 1e-3


def test_psnr_diff():
    a = torch.zeros(1, 3, 64, 64)
    b = torch.ones(1, 3, 64, 64)
    p = psnr(a, b)
    assert p.item() < 1.0


def test_lpips_smoke():
    a = torch.rand(1, 3, 64, 64) * 2 - 1
    b = a.clone()
    d = lpips_dist(a, b)
    assert d.item() < 0.05
