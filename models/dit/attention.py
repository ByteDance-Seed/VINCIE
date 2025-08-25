# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates  
# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn.functional as F

from common.mfu.utils import get_device_infos

try:
    gpu_type, _ = get_device_infos()
    assert gpu_type == "H800"
    from flash_attn_hopper import flash_attn_varlen_func

    USE_FA3 = True
except Exception:
    from flash_attn import flash_attn_varlen_func

    USE_FA3 = False

from torch import nn

from common.mfu import CustomFlops


class TorchAttention(nn.Module, CustomFlops):
    def tflops(self, args, kwargs, output) -> float:
        assert len(args) == 0 or len(args) > 2, "query, key should both provided by args / kwargs"
        q = kwargs.get("query") or args[0]
        k = kwargs.get("key") or args[1]
        b, h, sq, d = q.shape
        b, h, sk, d = k.shape
        return b * h * (4 * d * (sq / 1e6) * (sk / 1e6))

    def forward(self, *args, **kwargs):
        return F.scaled_dot_product_attention(*args, **kwargs)


class FlashAttentionVarlen(nn.Module, CustomFlops):
    def tflops(self, args, kwargs, output) -> float:
        cu_seqlens_q = kwargs["cu_seqlens_q"]
        cu_seqlens_k = kwargs["cu_seqlens_k"]
        _, h, d = output.shape
        seqlens_q = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]) / 1e6
        seqlens_k = (cu_seqlens_k[1:] - cu_seqlens_k[:-1]) / 1e6
        return h * (4 * d * (seqlens_q * seqlens_k).sum())

    def forward(self, *args, **kwargs):
        kwargs["deterministic"] = torch.are_deterministic_algorithms_enabled()
        return flash_attn_varlen_func(*args, **kwargs)
