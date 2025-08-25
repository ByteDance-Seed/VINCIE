# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates  
# SPDX-License-Identifier: Apache-2.0
from torch.nn.modules.utils import _triple

from .conv import CausalConv, CausalConvTranspose
from .types import _direction_t, _inflation_t, _pad_t, _size_3_t


class CausalDownsample(CausalConv):
    def __init__(
        self,
        nd: int,
        in_channels: int,
        out_channels: int,
        factor: _size_3_t,
        direction: _direction_t,
        inflation: _inflation_t,
        causal_pad: _pad_t,
    ):
        super().__init__(
            nd=nd,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=[max(2 * x, 3) for x in _triple(factor)],
            stride=factor,
            direction=direction,
            inflation=inflation,
            causal_pad=causal_pad,
        )


class CausalUpsample(CausalConvTranspose):
    def __init__(
        self,
        nd: int,
        in_channels: int,
        out_channels: int,
        factor: _size_3_t,
        direction: _direction_t,
        inflation: _inflation_t,
        causal_pad: _pad_t,
    ):
        super().__init__(
            nd=nd,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=[max(2 * x, 3) for x in _triple(factor)],
            stride=factor,
            direction=direction,
            inflation=inflation,
            causal_pad=causal_pad,
        )
