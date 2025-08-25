# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates  
# SPDX-License-Identifier: Apache-2.0

from .mmattn import NaMMAttention



attns = {
    "mm_full": NaMMAttention,
}


def get_attn(attn_type: str):
    if attn_type in attns:
        return attns[attn_type]
    raise NotImplementedError(f"{attn_type} is not supported")
