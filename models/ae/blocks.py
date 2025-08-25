# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates  
# SPDX-License-Identifier: Apache-2.0
from typing import Tuple
from torch import nn
from torch.nn.modules.utils import _triple

from .conv import CausalResBlock
from .resample import CausalDownsample, CausalUpsample
from .sequential import CausalSequential
from .types import (
    _activation_t,
    _direction_t,
    _inflation_t,
    _memory_t,
    _norm_t,
    _pad_t,
    _size_3_t,
    _tensor_t,
)


class CausalEncoderBlock(nn.Module):
    def __init__(
        self,
        *,
        nd: int,
        in_channels: int,
        out_channels: int,
        num_layers: int,
        downsample_factor: _size_3_t,
        direction: _direction_t,
        inflation: _inflation_t,
        causal_pad: _pad_t,
        norm: _norm_t,
        activation: _activation_t,
        skip_conv: bool,
        checkpointing: bool,
    ):
        super().__init__()
        if _triple(downsample_factor) != (1, 1, 1) or in_channels != out_channels:
            self.proj = CausalDownsample(
                nd=nd,
                in_channels=in_channels,
                out_channels=out_channels,
                factor=downsample_factor,
                direction=direction,
                inflation=inflation,
                causal_pad=causal_pad,
            )
        self.body = CausalSequential(
            modules=[
                CausalResBlock(
                    nd=nd,
                    in_channels=out_channels,
                    out_channels=out_channels,
                    direction=direction,
                    inflation=inflation,
                    causal_pad=causal_pad,
                    norm=norm,
                    activation=activation,
                    skip_conv=skip_conv,
                )
                for _ in range(num_layers)
            ],
            checkpointing=checkpointing,
        )

    def forward(
        self,
        x: _tensor_t,
        m: _memory_t,
    ) -> Tuple[
        _tensor_t,
        _memory_t,
    ]:
        m = m or {}
        if hasattr(self, "proj"):
            x, m["proj"] = self.proj(x, m.get("proj"))
        x, m["body"] = self.body(x, m.get("body"))
        return x, m


class CausalDecoderBlock(nn.Module):
    def __init__(
        self,
        *,
        nd: int,
        in_channels: int,
        out_channels: int,
        num_layers: int,
        upsample_factor: _size_3_t,
        direction: _direction_t,
        inflation: _inflation_t,
        causal_pad: _pad_t,
        norm: _norm_t,
        activation: _activation_t,
        skip_conv: bool,
        checkpointing: bool,
    ):
        super().__init__()
        self.body = CausalSequential(
            modules=[
                CausalResBlock(
                    nd=nd,
                    in_channels=in_channels,
                    out_channels=in_channels,
                    direction=direction,
                    inflation=inflation,
                    causal_pad=causal_pad,
                    norm=norm,
                    activation=activation,
                    skip_conv=skip_conv,
                )
                for _ in range(num_layers)
            ],
            checkpointing=checkpointing,
        )
        if _triple(upsample_factor) != (1, 1, 1) or in_channels != out_channels:
            self.proj = CausalUpsample(
                nd=nd,
                in_channels=in_channels,
                out_channels=out_channels,
                factor=upsample_factor,
                direction=direction,
                inflation=inflation,
                causal_pad=causal_pad,
            )

    def forward(
        self,
        x: _tensor_t,
        m: _memory_t,
    ) -> Tuple[
        _tensor_t,
        _memory_t,
    ]:
        m = m or {}
        x, m["body"] = self.body(x, m.get("body"))
        if hasattr(self, "proj"):
            x, m["proj"] = self.proj(x, m.get("proj"))
        return x, m
