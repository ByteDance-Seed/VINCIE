# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from .attn import NaSelfAttention, NaSpaceAttention, NaTemporalAttention, NaWindowAttention
from .cross import NaCrossAttention
from .mmattn import NaMMAttention, NaMMSpaceAttention, NaMMWindowAttention
from .mmcross import NaMMCrossAttention

try:
    import turbox  # noqa: F401
    from .fused_mmattn import FusedNaMMAttention, FusedNaMMSpaceAttention, FusedNaMMWindowAttention

    fused_attns = {
        "fused_mm_full": FusedNaMMAttention,
        "fused_mm_space": FusedNaMMSpaceAttention,
        "fused_mm_window": FusedNaMMWindowAttention,
    }
except Exception:
    fused_attns = {
        "fused_mm_full": NaMMAttention,
        "fused_mm_space": NaMMSpaceAttention,
        "fused_mm_window": NaMMWindowAttention,
    }

attns = {
    "cross": NaCrossAttention,
    "full": NaSelfAttention,
    "mm_full": NaMMAttention,
    "mm_cross": NaMMCrossAttention,
    "mm_space": NaMMSpaceAttention,
    "mm_window": NaMMWindowAttention,
    "space": NaSpaceAttention,
    "temporal": NaTemporalAttention,
    "window": NaWindowAttention,
    **fused_attns,
}


def get_attn(attn_type: str):
    if attn_type in attns:
        return attns[attn_type]
    raise NotImplementedError(f"{attn_type} is not supported")
