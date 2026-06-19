# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn.functional as F

from common.mfu.utils import get_device_infos


def _sdpa_varlen_fallback(q, k, v, cu_seqlens_q, cu_seqlens_k,
                           max_seqlen_q, max_seqlen_k,
                           dropout_p=0.0, softmax_scale=None,
                           causal=False, **kwargs):
    """Pure-PyTorch SDPA replacement for flash_attn_varlen_func.

    Unpacks variable-length packed sequences, runs torch SDPA with a padding
    mask, then repacks.  q/k/v: (total, num_heads, head_dim).
    """
    batch = cu_seqlens_q.shape[0] - 1
    H, D = q.shape[1], q.shape[2]
    device, dtype = q.device, q.dtype
    scale = softmax_scale or (D ** -0.5)

    q_pad = q.new_zeros(batch, max_seqlen_q, H, D)
    k_pad = k.new_zeros(batch, max_seqlen_k, H, D)
    v_pad = v.new_zeros(batch, max_seqlen_k, H, D)
    # True = valid token (attend), False = padding (ignore)
    mask = torch.zeros(batch, 1, 1, max_seqlen_k, device=device, dtype=torch.bool)

    for i in range(batch):
        sq = (cu_seqlens_q[i + 1] - cu_seqlens_q[i]).item()
        sk = (cu_seqlens_k[i + 1] - cu_seqlens_k[i]).item()
        q_pad[i, :sq] = q[cu_seqlens_q[i]:cu_seqlens_q[i + 1]]
        k_pad[i, :sk] = k[cu_seqlens_k[i]:cu_seqlens_k[i + 1]]
        v_pad[i, :sk] = v[cu_seqlens_k[i]:cu_seqlens_k[i + 1]]
        mask[i, 0, 0, :sk] = True

    # SDPA expects (B, H, S, D)
    out = F.scaled_dot_product_attention(
        q_pad.transpose(1, 2),
        k_pad.transpose(1, 2),
        v_pad.transpose(1, 2),
        attn_mask=mask,
        dropout_p=dropout_p,
        scale=scale,
        is_causal=causal,
    ).transpose(1, 2)  # (B, max_q, H, D)

    out_packed = torch.zeros_like(q)
    for i in range(batch):
        sq = (cu_seqlens_q[i + 1] - cu_seqlens_q[i]).item()
        out_packed[cu_seqlens_q[i]:cu_seqlens_q[i + 1]] = out[i, :sq]
    return out_packed


try:
    gpu_type, _ = get_device_infos()
    assert gpu_type == "H800"
    from flash_attn_hopper import flash_attn_varlen_func

    USE_FA3 = True
except Exception:
    try:
        from flash_attn import flash_attn_varlen_func

        USE_FA3 = False
    except ImportError:
        flash_attn_varlen_func = _sdpa_varlen_fallback
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
