"""
Phase 1 stub: cross-attention is NOT implemented yet.
Phase 3 fills this in.
"""


def fused_window_cross_attention(*args, **kwargs):
    raise NotImplementedError(
        "Cross-attention CUDA kernel lands in Phase 3. "
        "v6 trainer continues using oss/sr/v6/cross_attention.py PyTorch path."
    )
