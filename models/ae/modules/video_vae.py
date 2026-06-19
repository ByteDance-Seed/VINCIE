# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from contextlib import nullcontext
from typing import Optional, Tuple
import torch
import torch.nn as nn
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from einops import rearrange

from models.ae.modules.causal_inflation_lib import (
    InflatedCausalConv3d,
    UpscaleCausalConv3d,
    causal_class_wrapper,
    clean_memory_bank,
    remove_head,
)
from models.ae.modules.image_vae import ResnetBlock2D
from models.ae.modules.types import (
    CausalAutoencoderOutput,
    CausalDecoderOutput,
    CausalEncoderOutput,
    MemoryState,
    _inflation_mode_t,
    _memory_device_t,
    _receptive_field_t,
)


class Upsample3D(nn.Module):
    """A 3D upsampling layer."""

    def __init__(
        self,
        channels: int,
        inflation_mode: _inflation_mode_t = "tail",
        spatial_ratio: int = 2,
        temporal_ratio: int = 2,
    ):
        super().__init__()
        self.channels = channels
        self.conv = InflatedCausalConv3d(
            self.channels, self.channels, kernel_size=3, padding=1, inflation_mode=inflation_mode
        )

        self.temporal_ratio = temporal_ratio
        self.spatial_ratio = spatial_ratio

        upscale_ratio = (self.spatial_ratio**2) * self.temporal_ratio
        self.upscale_conv = UpscaleCausalConv3d(
            self.channels,
            self.channels * upscale_ratio,
            kernel_size=3,
            padding=1,
            inflation_mode=inflation_mode,
        )

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        memory_state: MemoryState,
    ) -> torch.FloatTensor:
        assert hidden_states.shape[1] == self.channels

        hidden_states = self.upscale_conv(hidden_states, memory_state=memory_state)
        hidden_states = rearrange(
            hidden_states,
            "b (z x y c) f h w -> b c (f z) (h x) (w y)",
            x=self.spatial_ratio,
            y=self.spatial_ratio,
            z=self.temporal_ratio,
        )

        # [Overridden] For causal temporal conv
        if self.temporal_ratio > 1 and memory_state != MemoryState.ACTIVE:
            hidden_states = remove_head(hidden_states, times=self.temporal_ratio - 1)

        hidden_states = self.conv(hidden_states, memory_state=memory_state)

        return hidden_states


class Downsample3D(nn.Module):
    """A 3D downsampling layer."""

    def __init__(
        self,
        channels: int,
        inflation_mode: _inflation_mode_t = "tail",
        temporal_down: bool = False,
    ):
        super().__init__()
        self.channels = channels
        self.temporal_down = temporal_down

        self.temporal_ratio = 2 if temporal_down else 1
        self.spatial_ratio = 2

        self.temporal_kernel = 3
        self.spatial_kernel = 4

        self.conv = InflatedCausalConv3d(
            self.channels,
            self.channels,
            kernel_size=(self.temporal_kernel, self.spatial_kernel, self.spatial_kernel),
            stride=(self.temporal_ratio, self.spatial_ratio, self.spatial_ratio),
            padding=(1, 1, 1),
            inflation_mode=inflation_mode,
        )

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        memory_state: MemoryState,
    ) -> torch.FloatTensor:
        assert hidden_states.shape[1] == self.channels
        hidden_states = self.conv(hidden_states, memory_state=memory_state)
        return hidden_states


class ResnetBlock3D(ResnetBlock2D):
    def __init__(
        self,
        *args,
        inflation_mode: _inflation_mode_t = "tail",
        time_receptive_field: _receptive_field_t = "half",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.conv1 = InflatedCausalConv3d(
            self.in_channels,
            self.out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            inflation_mode=inflation_mode,
        )

        self.norm1 = causal_class_wrapper(
            nn.GroupNorm(
                num_groups=32,
                num_channels=self.in_channels,
                eps=1e-6,
                affine=True,
            )
        )

        self.conv2 = InflatedCausalConv3d(
            self.out_channels,
            self.out_channels,
            kernel_size=(1, 3, 3) if time_receptive_field == "half" else (3, 3, 3),
            stride=1,
            padding=(0, 1, 1) if time_receptive_field == "half" else (1, 1, 1),
            inflation_mode=inflation_mode,
        )

        self.norm2 = causal_class_wrapper(
            nn.GroupNorm(
                num_groups=32,
                num_channels=self.out_channels,
                eps=1e-6,
                affine=True,
            )
        )

        if self.use_in_shortcut:
            self.conv_shortcut = InflatedCausalConv3d(
                self.in_channels,
                self.out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=(self.conv_shortcut.bias is not None),
                inflation_mode=inflation_mode,
            )

    def forward(self, input_tensor: torch.Tensor, memory_state: MemoryState = MemoryState.UNSET):
        assert memory_state != MemoryState.UNSET
        hidden = input_tensor

        hidden = self.norm1(hidden)
        hidden = self.nonlinearity(hidden)
        hidden = self.conv1(hidden, memory_state=memory_state)

        hidden = self.norm2(hidden)
        hidden = self.nonlinearity(hidden)
        hidden = self.dropout(hidden)
        hidden = self.conv2(hidden, memory_state=memory_state)

        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor, memory_state=memory_state)

        output_tensor = input_tensor + hidden

        return output_tensor


class DownEncoderBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        inflation_mode: _inflation_mode_t = "tail",
        time_receptive_field: _receptive_field_t = "half",
        temporal_down: bool = True,
    ):
        super().__init__()
        resnets = []

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock3D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    dropout=dropout,
                    inflation_mode=inflation_mode,
                    time_receptive_field=time_receptive_field,
                )
            )

        self.resnets = nn.ModuleList(resnets)

        self.downsampler = Downsample3D(
            channels=out_channels, inflation_mode=inflation_mode, temporal_down=temporal_down
        )

    def forward(
        self, hidden_states: torch.FloatTensor, memory_state: MemoryState
    ) -> torch.FloatTensor:
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, memory_state=memory_state)

        hidden_states = self.downsampler(hidden_states, memory_state=memory_state)

        return hidden_states


class FinalBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int,
        dropout: float = 0.0,
        inflation_mode: _inflation_mode_t = "tail",
        time_receptive_field: _receptive_field_t = "half",
    ):
        super().__init__()
        resnets = []
        for i in range(num_layers):
            input_channels = in_channels if i == 0 else out_channels

            resnets.append(
                ResnetBlock3D(
                    in_channels=input_channels,
                    out_channels=out_channels,
                    dropout=dropout,
                    inflation_mode=inflation_mode,
                    time_receptive_field=time_receptive_field,
                )
            )
        self.resnets = nn.ModuleList(resnets)

    def forward(self, hidden_states: torch.Tensor, memory_state: MemoryState):
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, memory_state=memory_state)
        return hidden_states


class UpDecoderBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        inflation_mode: _inflation_mode_t = "tail",
        time_receptive_field: _receptive_field_t = "half",
        temporal_up: bool = True,
    ):
        super().__init__()
        resnets = []

        for i in range(num_layers):
            input_channels = in_channels if i == 0 else out_channels

            resnets.append(
                ResnetBlock3D(
                    in_channels=input_channels,
                    out_channels=out_channels,
                    dropout=dropout,
                    inflation_mode=inflation_mode,
                    time_receptive_field=time_receptive_field,
                )
            )

        self.resnets = nn.ModuleList(resnets)

        self.upsampler = Upsample3D(
            channels=out_channels,
            inflation_mode=inflation_mode,
            spatial_ratio=2,
            temporal_ratio=2 if temporal_up else 1,
        )

    def forward(
        self, hidden_states: torch.FloatTensor, memory_state: MemoryState
    ) -> torch.FloatTensor:
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, memory_state=memory_state)

        hidden_states = self.upsampler(hidden_states, memory_state=memory_state)

        return hidden_states


class Encoder3D(nn.Module):
    r"""
    The `Encoder` layer of a variational autoencoder that encodes
    its input into a latent representation.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        block_out_channels: Tuple[int, ...] = (64,),
        layers_per_block: int = 2,
        double_z: bool = True,
        temporal_down_num: int = 2,
        gradient_checkpointing: bool = False,
        inflation_mode: _inflation_mode_t = "tail",
        time_receptive_field: _receptive_field_t = "half",
    ):
        super().__init__()
        self.layers_per_block = layers_per_block

        self.temporal_down_num = temporal_down_num

        self.conv_in = InflatedCausalConv3d(
            in_channels,
            block_out_channels[0],
            kernel_size=3,
            stride=1,
            padding=1,
            inflation_mode=inflation_mode,
        )

        self.down_blocks = nn.ModuleList([])

        # down
        output_channel = block_out_channels[0]
        for i in range(len(block_out_channels) - 1):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_temporal_down_block = i >= len(block_out_channels) - self.temporal_down_num - 1
            # Note: take the last one

            down_block = DownEncoderBlock3D(
                num_layers=self.layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                temporal_down=is_temporal_down_block,
                inflation_mode=inflation_mode,
                time_receptive_field=time_receptive_field,
            )
            self.down_blocks.append(down_block)

        # final
        self.final_block = FinalBlock3D(
            in_channels=output_channel,
            out_channels=block_out_channels[-1],
            num_layers=self.layers_per_block,
            inflation_mode=inflation_mode,
            time_receptive_field=time_receptive_field,
        )

        # out
        self.conv_norm_out = causal_class_wrapper(
            nn.GroupNorm(
                num_groups=32,
                num_channels=block_out_channels[-1],
                eps=1e-6,
            )
        )
        self.conv_act = nn.SiLU()

        conv_out_channels = 2 * out_channels if double_z else out_channels
        self.conv_out = InflatedCausalConv3d(
            block_out_channels[-1], conv_out_channels, 3, padding=1, inflation_mode=inflation_mode
        )

        self.gradient_checkpointing = gradient_checkpointing

    def forward(self, sample: torch.FloatTensor, memory_state: MemoryState) -> torch.FloatTensor:
        r"""The forward method of the `Encoder` class."""
        sample = self.conv_in(sample, memory_state=memory_state)
        if self.training and self.gradient_checkpointing:

            # down
            for down_block in self.down_blocks:
                sample = torch.utils.checkpoint.checkpoint(
                    down_block, sample, memory_state, use_reentrant=False
                )

            # final
            sample = torch.utils.checkpoint.checkpoint(
                self.mid_block, sample, memory_state, use_reentrant=False
            )

        else:
            # down
            for down_block in self.down_blocks:
                sample = down_block(sample, memory_state=memory_state)

            # final
            sample = self.final_block(sample, memory_state=memory_state)

        # post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample, memory_state=memory_state)

        return sample


class Decoder3D(nn.Module):
    r"""
    The `Decoder` layer of a variational autoencoder that
    decodes its latent representation into an output sample.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        block_out_channels: Tuple[int, ...] = (64,),
        layers_per_block: int = 2,
        inflation_mode: _inflation_mode_t = "tail",
        time_receptive_field: _receptive_field_t = "half",
        temporal_up_num: int = 2,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.layers_per_block = layers_per_block
        self.temporal_up_num = temporal_up_num

        self.conv_in = InflatedCausalConv3d(
            in_channels,
            block_out_channels[0],
            kernel_size=3,
            stride=1,
            padding=1,
            inflation_mode=inflation_mode,
        )

        self.up_blocks = nn.ModuleList([])

        # up
        output_channel = block_out_channels[0]
        for i in range(len(block_out_channels) - 1):
            prev_output_channel = output_channel
            output_channel = block_out_channels[i]

            is_temporal_up_block = i < self.temporal_up_num
            # Note: Keep symmetric

            up_block = UpDecoderBlock3D(
                num_layers=self.layers_per_block,
                in_channels=prev_output_channel,
                out_channels=output_channel,
                temporal_up=is_temporal_up_block,
                inflation_mode=inflation_mode,
                time_receptive_field=time_receptive_field,
            )
            self.up_blocks.append(up_block)

        # final
        self.final_block = FinalBlock3D(
            in_channels=output_channel,
            out_channels=block_out_channels[-1],
            num_layers=self.layers_per_block,
            inflation_mode=inflation_mode,
            time_receptive_field=time_receptive_field,
        )

        # out
        self.conv_norm_out = causal_class_wrapper(
            nn.GroupNorm(
                num_groups=32,
                num_channels=block_out_channels[-1],
                eps=1e-6,
            )
        )
        self.conv_act = nn.SiLU()
        self.conv_out = InflatedCausalConv3d(
            block_out_channels[-1], out_channels, 3, padding=1, inflation_mode=inflation_mode
        )

        self.gradient_checkpointing = gradient_checkpointing

    def forward(self, sample: torch.FloatTensor, memory_state: MemoryState) -> torch.FloatTensor:
        r"""The forward method of the `Decoder` class."""

        sample = self.conv_in(sample, memory_state=memory_state)

        if self.training and self.gradient_checkpointing:
            # up
            for up_block in self.up_blocks:
                sample = torch.utils.checkpoint.checkpoint(
                    up_block, sample, memory_state, use_reentrant=False
                )

            # final
            sample = torch.utils.checkpoint.checkpoint(
                self.final_block, sample, memory_state, use_reentrant=False
            )

        else:
            # up
            for up_block in self.up_blocks:
                sample = up_block(sample, memory_state=memory_state)

            # final
            sample = self.final_block(sample, memory_state=memory_state)

        # post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample, memory_state=memory_state)

        return sample


class VideoAutoencoderKL(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        enc_block_out_channels: Tuple[int] = (64,),
        dec_block_out_channels: Tuple[int] = (64,),
        enc_layers_per_block: int = 1,
        dec_layers_per_block: int = 1,
        latent_channels: int = 4,
        use_quant_conv: bool = True,
        use_post_quant_conv: bool = True,
        temporal_scale_num: int = 2,
        inflation_mode: _inflation_mode_t = "tail",
        enc_time_receptive_field: _receptive_field_t = "full",
        dec_time_receptive_field: _receptive_field_t = "full",
        slicing_sample_min_size: int = 4,
        spatial_downsample_factor: int = 16,
        temporal_downsample_factor: int = 4,
        freeze_encoder: bool = False,
    ):
        super().__init__()
        assert (2 ** (len(enc_block_out_channels) - 1)) == spatial_downsample_factor
        assert (2 ** (len(dec_block_out_channels) - 1)) == spatial_downsample_factor
        assert 2**temporal_scale_num == temporal_downsample_factor
        assert slicing_sample_min_size % temporal_downsample_factor == 0

        self.spatial_downsample_factor = spatial_downsample_factor
        self.temporal_downsample_factor = temporal_downsample_factor
        self.freeze_encoder = freeze_encoder
        self.slicing_sample_min_size = slicing_sample_min_size
        self.slicing_latent_min_size = slicing_sample_min_size // (2**temporal_scale_num)

        # pass init params to Encoder
        self.encoder = Encoder3D(
            in_channels=in_channels,
            out_channels=latent_channels,
            block_out_channels=enc_block_out_channels,
            layers_per_block=enc_layers_per_block,
            double_z=True,
            temporal_down_num=temporal_scale_num,
            gradient_checkpointing=False,
            inflation_mode=inflation_mode,
            time_receptive_field=enc_time_receptive_field,
        )

        # pass init params to Decoder
        self.decoder = Decoder3D(
            in_channels=latent_channels,
            out_channels=out_channels,
            block_out_channels=dec_block_out_channels,
            layers_per_block=dec_layers_per_block,
            # [Override] add temporal_up_num parameter
            temporal_up_num=temporal_scale_num,
            gradient_checkpointing=False,
            inflation_mode=inflation_mode,
            time_receptive_field=dec_time_receptive_field,
        )

        self.quant_conv = (
            InflatedCausalConv3d(
                in_channels=2 * latent_channels,
                out_channels=2 * latent_channels,
                kernel_size=1,
                inflation_mode=inflation_mode,
            )
            if use_quant_conv
            else None
        )
        self.post_quant_conv = (
            InflatedCausalConv3d(
                in_channels=latent_channels,
                out_channels=latent_channels,
                kernel_size=1,
                inflation_mode=inflation_mode,
            )
            if use_post_quant_conv
            else None
        )

        self.use_slicing = False

    def enable_slicing(self):
        self.use_slicing = True

    def disable_slicing(self):
        self.use_slicing = False

    def encode(self, x: torch.FloatTensor) -> CausalEncoderOutput:
        if x.ndim == 4:
            x = x.unsqueeze(2)
        h = self.slicing_encode(x)
        p = DiagonalGaussianDistribution(h)
        z = p.sample()
        return CausalEncoderOutput(z, p)

    def decode(self, z: torch.FloatTensor) -> CausalDecoderOutput:
        if z.ndim == 4:
            z = z.unsqueeze(2)
        x = self.slicing_decode(z)
        return CausalDecoderOutput(x)

    def _encode(self, x: torch.Tensor, memory_state: MemoryState) -> torch.Tensor:
        h = self.encoder(x, memory_state=memory_state)
        h = self.quant_conv(h, memory_state=memory_state) if self.quant_conv is not None else h
        return h

    def _decode(self, z: torch.Tensor, memory_state: MemoryState) -> torch.Tensor:
        z = (
            self.post_quant_conv(z, memory_state=memory_state)
            if self.post_quant_conv is not None
            else z
        )
        x = self.decoder(z, memory_state=memory_state)
        return x

    def slicing_encode(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_slicing and (x.shape[2] - 1) > self.slicing_sample_min_size:
            x_slices = x[:, :, 1:].split(split_size=self.slicing_sample_min_size, dim=2)
            encoded_slices = [
                self._encode(
                    torch.cat((x[:, :, :1], x_slices[0]), dim=2),
                    memory_state=MemoryState.INITIALIZING,
                )
            ]
            for x_idx in range(1, len(x_slices)):
                encoded_slices.append(
                    self._encode(x_slices[x_idx], memory_state=MemoryState.ACTIVE)
                )
            clean_memory_bank(self.encoder)
            clean_memory_bank(self.quant_conv)
            return torch.cat(encoded_slices, dim=2)
        else:
            return self._encode(x, memory_state=MemoryState.DISABLED)

    def slicing_decode(self, z: torch.Tensor) -> torch.Tensor:
        if self.use_slicing and (z.shape[2] - 1) > self.slicing_latent_min_size:
            z_slices = z[:, :, 1:].split(split_size=self.slicing_latent_min_size, dim=2)
            decoded_slices = [
                self._decode(
                    torch.cat((z[:, :, :1], z_slices[0]), dim=2),
                    memory_state=MemoryState.INITIALIZING,
                )
            ]
            for z_idx in range(1, len(z_slices)):
                decoded_slices.append(
                    self._decode(z_slices[z_idx], memory_state=MemoryState.ACTIVE)
                )
            clean_memory_bank(self.post_quant_conv)
            clean_memory_bank(self.decoder)
            return torch.cat(decoded_slices, dim=2)
        else:
            return self._decode(z, memory_state=MemoryState.DISABLED)

    def forward(self, x: torch.FloatTensor) -> CausalAutoencoderOutput:
        with torch.no_grad() if self.freeze_encoder else nullcontext():
            z, p = self.encode(x)
        assert x.size(-2) // z.size(-2) == self.spatial_downsample_factor
        assert x.size(-1) // z.size(-1) == self.spatial_downsample_factor
        assert (z.size(2) == 1 and (x.ndim == 4 or x.size(2) == 1)) or (
            (x.size(2) - 1) // (z.size(2) - 1) == self.temporal_downsample_factor
        )
        x = self.decode(z).sample
        return CausalAutoencoderOutput(x, z, p)

    def preprocess(self, x: torch.Tensor):
        # x should in [B, C, T, H, W], [B, C, H, W]
        assert x.ndim == 4 or x.size(2) % self.temporal_downsample_factor == 1
        return x

    def postprocess(self, x: torch.Tensor):
        # x should in [B, C, T, H, W], [B, C, H, W]
        return x

    def set_causal_slicing(
        self,
        *,
        split_size: Optional[int],
        memory_device: _memory_device_t,
    ):
        assert (
            split_size is None or memory_device is not None
        ), "if split_size is set, memory_device must not be None."
        if split_size is not None:
            self.enable_slicing()
            self.slicing_sample_min_size = split_size
            self.slicing_latent_min_size = split_size // self.temporal_downsample_factor
        else:
            self.disable_slicing()
        for module in self.modules():
            if isinstance(module, InflatedCausalConv3d):
                module.set_memory_device(memory_device)
