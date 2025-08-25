# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates  
# SPDX-License-Identifier: Apache-2.0
from einops import rearrange
from torch import nn

from .types import _norm_t, _tensor_t


def get_norm_layer(norm_type: _norm_t) -> nn.Module:

    def _norm_layer(channels: int):
        if norm_type is None:
            return nn.Identity()
        if norm_type == "group":
            return CausalGroupNorm(32, channels)
        raise NotImplementedError

    return _norm_layer


class CausalGroupNorm(nn.GroupNorm):
    def forward(self, x: _tensor_t) -> _tensor_t:
        if x.ndim <= 4:
            return super().forward(x)
        if x.ndim == 5:
            # must not compute across the temporal dimension to ensure causality.
            t = x.size(2)
            x = rearrange(x, "b c t h w -> (b t) c h w")
            x = super().forward(x)
            x = rearrange(x, "(b t) c h w -> b c t h w", t=t)
            return x
        raise NotImplementedError
