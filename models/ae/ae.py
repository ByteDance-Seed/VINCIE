# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates  
# SPDX-License-Identifier: Apache-2.0
import math
from typing import Callable, Literal, NamedTuple, Optional, Sequence
import torch
from torch import nn

from .act import get_act_layer
from .blocks import CausalDecoderBlock, CausalEncoderBlock
from .conv import CausalConv, Conv
from .distribution import DiagonalGaussianDistribution
from .norm import get_norm_layer
from .pad import CausalPad, pad
from .sequential import CausalSequential
from .types import _activation_t, _direction_t, _inflation_t, _memory_t, _norm_t, _pad_t, _tensor_t


class CausalAutoencoderOutput(NamedTuple):
    sample: _tensor_t
    latent: _tensor_t
    memory: _memory_t
    posterior: Optional[DiagonalGaussianDistribution]


class CausalEncoderOutput(NamedTuple):
    latent: _tensor_t
    memory: _memory_t
    posterior: Optional[DiagonalGaussianDistribution]


class CausalDecoderOutput(NamedTuple):
    sample: _tensor_t
    memory: _memory_t


class CausalAutoencoder(nn.Module):
    def __init__(
        self,
        *,
        nd: int = 3,
        in_channels: int = 3,
        out_channels: int = 3,
        latent_channels: int = 16,
        norm: _norm_t = "group",
        activation: _activation_t = "silu",
        skip_conv: bool = False,
        output_norm_act: bool = True,
        variational: bool = False,
        inflation: _inflation_t = "tail",
        spatial_downsample_factor: int = 8,
        temporal_downsample_factor: int = 4,
        encoder_channels: Sequence[int] = (128, 256, 256, 512),
        encoder_num_layers: Sequence[int] = (4, 4, 4, 8),
        encoder_spatial_downsample: Sequence[int] = (1, 2, 2, 2),
        encoder_temporal_downsample: Sequence[int] = (1, 1, 2, 2),
        encoder_direction: _direction_t = "forward",
        encoder_causal_pad: _pad_t = "constant",
        decoder_channels: Sequence[int] = (128, 256, 256, 512),
        decoder_num_layers: Sequence[int] = (4, 4, 4, 8),
        decoder_spatial_upsample: Sequence[int] = (1, 2, 2, 2),
        decoder_temporal_upsample: Sequence[int] = (1, 1, 2, 2),
        decoder_direction: _direction_t = "forward",
        decoder_causal_pad: _pad_t = "constant",
        checkpointing: bool = False,
    ):
        assert spatial_downsample_factor == math.prod(encoder_spatial_downsample)
        assert spatial_downsample_factor == math.prod(decoder_spatial_upsample)
        assert temporal_downsample_factor == math.prod(encoder_temporal_downsample)
        assert temporal_downsample_factor == math.prod(decoder_temporal_upsample)
        super().__init__()
        self.temporal_downsample_factor = temporal_downsample_factor
        self.encoder_causal_pad = encoder_causal_pad
        self.encoder_direction = encoder_direction
        self.encoder = CausalEncoder(
            nd=nd,
            in_channels=in_channels,
            latent_channels=latent_channels,
            channels=encoder_channels,
            num_layers=encoder_num_layers,
            spatial_downsample=encoder_spatial_downsample,
            temporal_downsample=encoder_temporal_downsample,
            direction=encoder_direction,
            inflation=inflation,
            causal_pad=encoder_causal_pad,
            norm=norm,
            activation=activation,
            skip_conv=skip_conv,
            output_norm_act=output_norm_act,
            variational=variational,
            checkpointing=checkpointing,
        )
        self.decoder = CausalDecoder(
            nd=nd,
            out_channels=out_channels,
            latent_channels=latent_channels,
            channels=decoder_channels,
            num_layers=decoder_num_layers,
            spatial_upsample=decoder_spatial_upsample,
            temporal_upsample=decoder_temporal_upsample,
            direction=decoder_direction,
            inflation=inflation,
            causal_pad=decoder_causal_pad,
            norm=norm,
            activation=activation,
            skip_conv=skip_conv,
            output_norm_act=output_norm_act,
            checkpointing=checkpointing,
        )

    def forward(
        self,
        x: _tensor_t,
        m: _memory_t = None,
    ) -> CausalAutoencoderOutput:
        m = m or {}
        z, m["encoder"], p = self.encoder(x, m.get("encoder"))
        x, m["decoder"] = self.decoder(z, m.get("decoder"))
        return CausalAutoencoderOutput(x, z, m, p)

    def encode(
        self,
        x: _tensor_t,
        m: _memory_t = None,
    ) -> CausalEncoderOutput:
        m = m or {}
        z, m["encoder"], p = self.encoder(x, m.get("encoder"))
        return CausalEncoderOutput(z, m, p)

    def decode(
        self,
        z: _tensor_t,
        m: _memory_t = None,
    ) -> CausalDecoderOutput:
        m = m or {}
        x, m["decoder"] = self.decoder(z, m.get("decoder"))
        return CausalDecoderOutput(x, m)

    def preprocess(
        self,
        x: _tensor_t,
    ) -> _tensor_t:
        if x.ndim == 5:
            # Pad to make the first frame occupy a dedicated latent.
            pt = self.temporal_downsample_factor - 1
            x = pad(x, pt, mode=self.encoder_causal_pad, direction=self.encoder_direction)
            assert x.size(2) % self.temporal_downsample_factor == 0
        return x

    def postprocess(
        self,
        x: _tensor_t,
    ) -> _tensor_t:
        if x.ndim == 5:
            # Unpad to remove the duplicated frames.
            assert x.size(2) % self.temporal_downsample_factor == 0
            pt = self.temporal_downsample_factor - 1
            x = pad(x, -pt, mode=self.encoder_causal_pad, direction=self.encoder_direction)
        return x

    def set_causal_slicing(
        self,
        *,
        split_size: Optional[int],
        memory_device: Optional[Literal["cpu", "same"]],
    ):
        assert (
            split_size is None or memory_device is not None
        ), "if split_size is set, memory_device must not be None."
        self.encoder.temporal_split_size = split_size
        self.decoder.temporal_split_size = split_size
        for module in self.modules():
            if isinstance(module, CausalPad):
                module.set_memory_device(memory_device)


def forward_split(
    x: _tensor_t,
    m: _memory_t,
    size: int,
    func: Callable,
    direction: _direction_t,
    cls: Callable,
):
    if x.ndim == 4 or size is None:
        return func(x, m)
    else:
        x = x.split(size, dim=2)
        if direction == "backward":
            x = list(reversed(x))
        o = []
        for f in x:
            f, m, *args = func(f, m)
            o.append(f)
        if direction == "backward":
            o = list(reversed(o))
        o = torch.cat(o, dim=2)
        return cls(o, m, *args)


class CausalEncoder(nn.Module):
    def __init__(
        self,
        *,
        nd: int = 3,
        in_channels: int = 3,
        latent_channels: int = 16,
        channels: Sequence[int] = (128, 256, 256, 512),
        num_layers: Sequence[int] = (4, 4, 4, 8),
        spatial_downsample: Sequence[int] = (1, 2, 2, 2),
        temporal_downsample: Sequence[int] = (1, 1, 2, 2),
        direction: _direction_t = "forward",
        inflation: _inflation_t = "tail",
        causal_pad: _pad_t = "constant",
        norm: _norm_t = "group",
        activation: _activation_t = "silu",
        skip_conv: bool = False,
        output_norm_act: bool = True,
        variational: bool = False,
        checkpointing: bool = False,
    ):
        super().__init__()
        channels = tuple(channels[:1] + channels)
        self.temporal_split_size = None
        self.direction = direction
        self.input = CausalConv(
            nd=nd,
            in_channels=in_channels,
            out_channels=channels[0],
            kernel_size=3,
            direction=direction,
            inflation=inflation,
            causal_pad=causal_pad,
        )
        self.blocks = CausalSequential(
            [
                CausalEncoderBlock(
                    nd=nd,
                    in_channels=channels[i],
                    out_channels=channels[i + 1],
                    num_layers=num_layers[i],
                    direction=direction,
                    inflation=inflation,
                    causal_pad=causal_pad,
                    norm=norm,
                    activation=activation,
                    skip_conv=skip_conv,
                    checkpointing=checkpointing,
                    downsample_factor=(
                        temporal_downsample[i],
                        spatial_downsample[i],
                        spatial_downsample[i],
                    ),
                )
                for i in range(len(num_layers))
            ]
        )
        self.norm_act = (
            nn.Sequential(
                get_norm_layer(norm)(channels[-1]),
                get_act_layer(activation),
            )
            if output_norm_act
            else nn.Identity()
        )
        self.quantize = Conv(
            nd=nd,
            in_channels=channels[-1],
            out_channels=latent_channels,
            kernel_size=1,
            direction=direction,
            inflation=inflation,
            causal_pad=causal_pad,
        )
        self.variance = (
            Conv(
                nd=nd,
                in_channels=channels[-1],
                out_channels=latent_channels,
                kernel_size=1,
                direction=direction,
                inflation=inflation,
                causal_pad=causal_pad,
            )
            if variational
            else None
        )

    def forward(
        self,
        x: _tensor_t,
        m: _memory_t = None,
    ) -> CausalEncoderOutput:
        return forward_split(
            x=x,
            m=m,
            size=self.temporal_split_size,
            func=self.forward_internal,
            direction=self.direction,
            cls=CausalEncoderOutput,
        )

    def forward_internal(
        self,
        x: _tensor_t,
        m: _memory_t = None,
    ) -> CausalEncoderOutput:
        m = m or {}
        x, m["input"] = self.input(x, m.get("input"))
        x, m["blocks"] = self.blocks(x, m.get("blocks"))
        x = self.norm_act(x)
        z = self.quantize(x)

        if self.variance is not None:
            v = self.variance(x)
            p = DiagonalGaussianDistribution(z, v)
            z = p.sample()
        else:
            p = None

        return CausalEncoderOutput(z, m, p)


class CausalDecoder(nn.Module):
    def __init__(
        self,
        *,
        nd: int = 3,
        out_channels: int = 3,
        latent_channels: int = 16,
        channels: Sequence[int] = (128, 256, 256, 512),
        num_layers: Sequence[int] = (4, 4, 4, 8),
        spatial_upsample: Sequence[int] = (1, 2, 2, 2),
        temporal_upsample: Sequence[int] = (1, 1, 2, 2),
        direction: _direction_t = "forward",
        inflation: _inflation_t = "tail",
        causal_pad: _pad_t = "constant",
        norm: _norm_t = "group",
        activation: _activation_t = "silu",
        skip_conv: bool = False,
        output_norm_act: bool = True,
        checkpointing: bool = False,
    ):
        super().__init__()
        channels = tuple(channels[:1] + channels)
        self.temporal_split_size = None
        self.temporal_upsample_factor = math.prod(temporal_upsample)
        self.direction = direction
        self.dequantize = Conv(
            nd=nd,
            in_channels=latent_channels,
            out_channels=channels[-1],
            kernel_size=1,
            direction=direction,
            inflation=inflation,
            causal_pad=causal_pad,
        )
        self.blocks = CausalSequential(
            [
                CausalDecoderBlock(
                    nd=nd,
                    in_channels=channels[i + 1],
                    out_channels=channels[i],
                    num_layers=num_layers[i],
                    direction=direction,
                    inflation=inflation,
                    causal_pad=causal_pad,
                    norm=norm,
                    skip_conv=skip_conv,
                    checkpointing=checkpointing,
                    activation=activation,
                    upsample_factor=(
                        temporal_upsample[i],
                        spatial_upsample[i],
                        spatial_upsample[i],
                    ),
                )
                for i in reversed(range(len(num_layers)))
            ]
        )
        self.norm_act = (
            nn.Sequential(
                get_norm_layer(norm)(channels[0]),
                get_act_layer(activation),
            )
            if output_norm_act
            else nn.Identity()
        )
        self.output = CausalConv(
            nd=nd,
            in_channels=channels[0],
            out_channels=out_channels,
            kernel_size=3,
            direction=direction,
            inflation=inflation,
            causal_pad=causal_pad,
        )

    def forward(
        self,
        x: _tensor_t,
        m: _memory_t = None,
    ) -> CausalDecoderOutput:
        return forward_split(
            x=x,
            m=m,
            size=(
                (self.temporal_split_size // self.temporal_upsample_factor)
                if self.temporal_split_size is not None
                else None
            ),
            func=self.forward_internal,
            direction=self.direction,
            cls=CausalDecoderOutput,
        )

    def forward_internal(
        self,
        x: _tensor_t,
        m: _memory_t = None,
    ) -> CausalDecoderOutput:
        m = m or {}
        x = self.dequantize(x)
        x, m["blocks"] = self.blocks(x, m.get("blocks"))
        x = self.norm_act(x)
        x, m["output"] = self.output(x, m.get("output"))
        return CausalDecoderOutput(x, m)
