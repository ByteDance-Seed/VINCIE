# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates  
# SPDX-License-Identifier: Apache-2.0
from typing import Optional, Tuple
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _triple

from common.logger import get_logger
from common.mfu import CustomFlops
from common.mfu.basic_hooks import conv_tflops_func

from .act import get_act_layer
from .norm import get_norm_layer
from .pad import CausalPad
from .types import (
    _activation_t,
    _direction_t,
    _inflation_t,
    _memory_dict_t,
    _memory_tensor_t,
    _norm_t,
    _pad_t,
    _size_3_t,
    _tensor_t,
)

logger = get_logger(__name__)


class Conv(nn.Module, CustomFlops):
    def __init__(
        self,
        *,
        nd: int,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_3_t,
        stride: _size_3_t = 1,
        padding: _size_3_t = 0,
        bias: bool = True,
        transposed: bool = False,
        causal_pad: _pad_t,
        direction: _direction_t,
        inflation: _inflation_t,
        gain: float = 1.0,
    ):
        assert nd in [2, 3]
        super().__init__()
        self.nd = nd
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _triple(kernel_size)[-nd:]
        self.stride = _triple(stride)[-nd:]
        self.padding = _triple(padding)[-nd:]
        self.transposed = transposed
        self.causal_pad = causal_pad
        self.direction = direction
        self.inflation = inflation
        self.gain = gain
        self.channels = (
            (out_channels, in_channels) if not transposed else (in_channels, out_channels)
        )
        self.weight = nn.Parameter(torch.empty(*self.channels, *self.kernel_size))
        self.bias = nn.Parameter(torch.empty(out_channels)) if bias else None
        _ConvNd.reset_parameters(self)
        if self.gain != 1.0:
            self.weight.data *= self.gain
            if self.bias is not None:
                self.bias.data *= self.gain

    def tflops(self, args, kwargs, output) -> float:
        return conv_tflops_func(self, args, kwargs, output)

    def forward(self, x: _tensor_t) -> _tensor_t:
        w, b, s, p = self.weight, self.bias, self.stride, self.padding

        if x.ndim == 4 and w.ndim == 5:
            w, s, p = self.get_merged_weight(), s[-2:], p[-2:]

        if x.ndim == 4:
            conv = F.conv2d if not self.transposed else F.conv_transpose2d
        elif x.ndim == 5:
            conv = F.conv3d if not self.transposed else F.conv_transpose3d
        else:
            raise NotImplementedError

        return conv(x, w, b, s, p)

    def get_merged_weight(self):
        if not self.transposed:
            i = -1 if self.direction == "forward" else 0
            if self.causal_pad == "constant":
                return self.weight[:, :, i]
            if self.causal_pad == "replicate":
                return self.weight.sum(2)
        else:
            i = 0 if self.direction == "forward" else -1
            kt, st = self.kernel_size[0], self.stride[0]
            if self.causal_pad == "constant":
                return self.weight[:, :, i]
            if self.causal_pad == "replicate":
                return self.weight[:, :, (kt + i) % st :: st].sum(2)
        raise NotImplementedError

    @torch.no_grad()
    def inflate_weight_on_channel(self, weight: _tensor_t, div_term: float = 1e4) -> _tensor_t:
        src_ch = np.array(weight.size()[:2])
        tgt_ch = np.array(self.channels)
        if (src_ch == tgt_ch).all():
            return weight
        fan = tgt_ch / src_ch
        assert (
            fan >= 1.0
        ).all(), "The target channels should be all larger than the original ones."
        out = torch.randn(*tgt_ch, *weight.size()[2:]) / div_term
        out[: src_ch[0], : src_ch[1]] = weight  # Constant Mode By Default
        return out

    @torch.no_grad()
    def inflate_bias_on_channel(self, bias: _tensor_t, div_term: float = 1e4) -> _tensor_t:
        src_ch = bias.size(0)
        tgt_ch = self.bias.size(0)
        if src_ch == tgt_ch:
            return bias
        fan = tgt_ch / src_ch
        assert fan >= 1.0, "The target channels should be all larger than the original ones."
        out = torch.randn(tgt_ch) / div_term
        out[:src_ch] = bias  # Constant Mode By Default
        return out

    @torch.no_grad()
    def inflate_weight_on_time(self, weight: _tensor_t) -> _tensor_t:
        if weight.size() == self.weight.size():
            return weight
        assert weight.size()[-2:] == self.weight.size()[-2:]
        if weight.ndim == 4:
            weight = weight.unsqueeze(2)
        if not self.transposed:
            if self.inflation == "tail" or (self.weight.size(2) % weight.size(2) != 0):
                out = torch.zeros_like(self.weight.data).to(weight)
                if self.direction == "forward":
                    out[:, :, -weight.size(2) :] = weight
                else:
                    out[:, :, : weight.size(2)] = weight
            elif self.inflation == "random":
                ratio = torch.rand([1, 1, *self.weight.shape[2:]], device=weight.device)
                ratio = ratio / ratio.sum(2, keepdim=True)
                out = weight.expand_as(self.weight) * ratio
            else:
                raise NotImplementedError
        else:
            if self.inflation == "tail" or (self.weight.size(2) % weight.size(2) != 0):
                out = torch.zeros_like(self.weight.data).to(weight)
                if self.direction == "forward":
                    out[:, :, : weight.size(2)] = weight
                else:
                    out[:, :, -weight.size(2) :] = weight
            elif self.inflation == "random":
                st = self.stride[0]
                ratio = torch.rand([1, 1, *self.weight.shape[2:]], device=weight.device)
                for i in range(st):
                    ratio[:, :, i::st] /= ratio[:, :, i::st].sum(2, keepdim=True)
                out = weight.expand_as(self.weight) * ratio
            else:
                raise NotImplementedError
        return out

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        weight_name = prefix + "weight"
        bias_name = prefix + "bias"
        if weight_name in state_dict:
            weight = state_dict[weight_name]
            weight = self.inflate_weight_on_channel(weight)
            weight = self.inflate_weight_on_time(weight)
            state_dict[weight_name] = weight
        if bias_name in state_dict:
            bias = state_dict[bias_name]
            bias = self.inflate_bias_on_channel(bias)
            state_dict[bias_name] = bias
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def extra_repr(self) -> str:
        keys = [
            "nd",
            "in_channels",
            "out_channels",
            "kernel_size",
            "stride",
            "padding",
            "transposed",
            "causal_pad",
            "direction",
            "inflation",
            "gain",
        ]
        return ",\n".join([f"{key}={str(getattr(self, key))}" for key in keys])


class CausalConv(nn.Module):
    def __init__(
        self,
        nd: int,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_3_t,
        direction: _direction_t,
        inflation: _inflation_t,
        stride: _size_3_t = 1,
        padding: Optional[_size_3_t] = None,
        causal_pad: _pad_t = "constant",
        gain: float = 1.0,
    ):
        super().__init__()
        if padding is None:
            # Default to same padding.
            st, sh, sw = _triple(stride)
            kt, kh, kw = _triple(kernel_size)
            pt, ph, pw = kt - st, (kh - sh) // 2, (kw - sw) // 2
        else:
            pt, ph, pw = _triple(padding)

        self.cpad = CausalPad(
            size=pt,
            mode=causal_pad,
            direction=direction,
        )
        self.conv = Conv(
            nd=nd,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(0, ph, pw),
            direction=direction,
            inflation=inflation,
            causal_pad=causal_pad,
            gain=gain,
        )

    def forward(
        self,
        x: _tensor_t,
        m: _memory_tensor_t,
    ) -> Tuple[
        _tensor_t,
        _memory_tensor_t,
    ]:
        x, m = self.cpad(x, m)
        x = self.conv(x)
        return x, m


class CausalConvTranspose(nn.Module):
    def __init__(
        self,
        nd: int,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_3_t,
        direction: _direction_t,
        inflation: _inflation_t,
        stride: _size_3_t = 1,
        padding: Optional[_size_3_t] = None,
        causal_pad: _pad_t = "constant",
    ):
        super().__init__()
        if padding is None:
            # Default to same padding.
            st, sh, sw = _triple(stride)
            kt, kh, kw = _triple(kernel_size)
            pt, ph, pw = (kt - 1) // st, (kh - sh) // 2, (kw - sw) // 2
        else:
            pt, ph, pw = _triple(padding)

        self.cpad = CausalPad(
            size=pt,
            mode=causal_pad,
            direction=direction,
        )
        self.conv = Conv(
            nd=nd,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(pt * st, ph, pw),
            direction=direction,
            inflation=inflation,
            causal_pad=causal_pad,
            transposed=True,
        )

    def forward(
        self,
        x: _tensor_t,
        m: _memory_tensor_t,
    ) -> Tuple[
        _tensor_t,
        _memory_tensor_t,
    ]:
        x, m = self.cpad(x, m)
        x = self.conv(x)
        return x, m


class CausalResBlock(nn.Module):
    def __init__(
        self,
        nd: int,
        in_channels: int,
        out_channels: int,
        direction: _direction_t,
        inflation: _inflation_t,
        causal_pad: _pad_t,
        norm: _norm_t,
        activation: _activation_t,
        skip_conv: bool,
    ):
        super().__init__()
        skip_conv = skip_conv or in_channels != out_channels
        norm = get_norm_layer(norm)
        self.norm1 = norm(in_channels)
        self.conv1 = CausalConv(
            nd=nd,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            direction=direction,
            inflation=inflation,
            causal_pad=causal_pad,
        )
        self.norm2 = norm(out_channels)
        self.conv2 = CausalConv(
            nd=nd,
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            direction=direction,
            inflation=inflation,
            causal_pad=causal_pad,
            gain=0,
        )
        self.skip = (
            Conv(
                nd=nd,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                bias=False,
                direction=direction,
                inflation=inflation,
                causal_pad=causal_pad,
            )
            if skip_conv
            else None
        )
        self.act = get_act_layer(activation)

    def forward(
        self,
        x: _tensor_t,
        m: _memory_dict_t,
    ) -> Tuple[
        _tensor_t,
        _memory_dict_t,
    ]:
        m = m or {}
        y = self.skip(x) if self.skip else x

        x = self.norm1(x)
        x = self.act(x)
        x, m["conv1"] = self.conv1(x, m.get("conv1"))

        x = self.norm2(x)
        x = self.act(x)
        x, m["conv2"] = self.conv2(x, m.get("conv2"))

        o = x + y

        return o, m
