import torch

from oss.sr.v6.student import FasterNetBlock, StudentBackbone


def test_fasternet_block_identity_at_init():
    """conv_contract zero-init means block starts as identity."""
    block = FasterNetBlock(channels=48)
    x = torch.randn(1, 48, 16, 16)
    out = block(x)
    assert torch.allclose(out, x, atol=1e-6)


def test_student_backbone_param_count_under_1m():
    """Default-config student should be <=1.2M params (target ~1M)."""
    model = StudentBackbone(in_channels=9, channels=48, depth=4, out_features=180)
    n = model.num_params()
    assert n <= 1_200_000, f"student backbone too large: {n} params"
    print(f"StudentBackbone params: {n}")


def test_student_backbone_forward():
    model = StudentBackbone(in_channels=9, channels=48, depth=4, out_features=180)
    x = torch.randn(2, 9, 64, 96)
    out = model(x)
    assert out.shape == (2, 180, 64, 96)
