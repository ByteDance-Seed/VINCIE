 # Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
 # SPDX-License-Identifier: Apache-2.0 

"""
General tflop count equations for basic nn.Module
e.g. Linear, ConvNd, ..
Can register_forward_hook on module directly.
We convert flops -> tflops (/1e12) at the op-level to prevent value overflow
"""

import math
from typing import List, Tuple
from diffusers.models.attention_processor import Attention
from torch import nn
from torch.nn.modules.conv import _ConvNd
from transformers.models.t5.modeling_t5 import T5Attention


def conv_tflops_func(module, args, kwargs, output):
    return (2 * math.prod(module.kernel_size) * module.in_channels * (output.numel() / 1e6)) / 1e6


def linear_tflops_func(module, args, kwargs, output):
    return (2 * module.in_features * (output.numel() / 1e6)) / 1e6


def hf_self_attn_tflops_func(module, args, kwargs, output):
    output = output[0] if isinstance(output, (List, Tuple)) else output
    if output.ndim == 4:
        bsz, hid_dim, h, w = output.shape
        seq_len = h * w
    elif output.ndim == 3:
        bsz, seq_len, hid_dim = output.shape
    else:
        raise NotImplementedError()
    return bsz * (
        8 * hid_dim * hid_dim * (seq_len / 1e12) + 4 * (seq_len / 1e6) * (seq_len / 1e6) * hid_dim
    )


basic_flops_func = {
    _ConvNd: conv_tflops_func,
    nn.Linear: linear_tflops_func,
    Attention: hf_self_attn_tflops_func,
    T5Attention: hf_self_attn_tflops_func,
    "QWenAttention": hf_self_attn_tflops_func,
    "Qwen2FlashAttention2": hf_self_attn_tflops_func,
}
