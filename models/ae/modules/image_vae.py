from typing import Optional, Tuple
import torch
import torch.nn.functional as F
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from torch import nn

from models.ae.modules.types import (
    CausalAutoencoderOutput,
    CausalDecoderOutput,
    CausalEncoderOutput,
)


class ResnetBlock2D(nn.Module):
    r"""
    A Resnet block.

    Parameters:
        in_channels (`int`): The number of channels in the input.
        out_channels (`int`, *optional*, default to be `None`):
            The number of output channels for the first conv2d layer.
            If None, same as `in_channels`.
        dropout (`float`, *optional*, defaults to `0.0`): The dropout probability to use.
    """

    def __init__(
        self, *, in_channels: int, out_channels: Optional[int] = None, dropout: float = 0.0
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        self.nonlinearity = nn.SiLU()

        self.norm1 = torch.nn.GroupNorm(
            num_groups=32, num_channels=in_channels, eps=1e-6, affine=True
        )

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.norm2 = torch.nn.GroupNorm(
            num_groups=32, num_channels=out_channels, eps=1e-6, affine=True
        )

        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.use_in_shortcut = self.in_channels != out_channels

        self.conv_shortcut = None
        if self.use_in_shortcut:
            self.conv_shortcut = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=1, padding=0
            )

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden = input_tensor

        hidden = self.norm1(hidden)
        hidden = self.nonlinearity(hidden)
        hidden = self.conv1(hidden)

        hidden = self.norm2(hidden)
        hidden = self.nonlinearity(hidden)
        hidden = self.dropout(hidden)
        hidden = self.conv2(hidden)

        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor)

        output_tensor = input_tensor + hidden

        return output_tensor


class Downsample2D(nn.Module):
    """A 2D downsampling layer

    Parameters:
        channels (`int`): number of channels in the inputs and outputs.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        self.conv = nn.Conv2d(self.channels, self.channels, kernel_size=4, stride=2, padding=1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        assert hidden_states.shape[1] == self.channels
        hidden_states = self.conv(hidden_states)
        return hidden_states


class Upsample2D(nn.Module):
    """A 2D upsampling layer

    Parameters:
        channels (`int`): number of channels in the inputs and outputs.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        self.conv = nn.Conv2d(self.channels, self.channels, kernel_size=3, padding=1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        assert hidden_states.shape[1] == self.channels

        if hidden_states.shape[0] >= 64:
            hidden_states = hidden_states.contiguous()

        hidden_states = F.interpolate(hidden_states, scale_factor=2.0, mode="nearest")

        hidden_states = self.conv(hidden_states)

        return hidden_states


class DownEncoderBlock2D(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, dropout: float = 0.0, num_layers: int = 1
    ):
        super().__init__()
        resnets = []

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock2D(in_channels=in_channels, out_channels=out_channels, dropout=dropout)
            )

        self.resnets = nn.ModuleList(resnets)

        self.downsampler = Downsample2D(out_channels)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states)

        hidden_states = self.downsampler(hidden_states)

        return hidden_states


class FinalBlock2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_layers: int, dropout: float = 0.0):
        super().__init__()
        resnets = []

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock2D(in_channels=in_channels, out_channels=out_channels, dropout=dropout)
            )

        self.resnets = nn.ModuleList(resnets)

    def forward(self, hidden_states: torch.Tensor):
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states)
        return hidden_states


class Encoder2D(nn.Module):
    r"""
    The `Encoder` layer of a variational autoencoder
        that encodes its input into a latent representation.

    Args:
        in_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        out_channels (`int`, *optional*, defaults to 3):
            The number of output channels.
        block_out_channels (`Tuple[int, ...]`, *optional*, defaults to `(64,)`):
            The number of output channels for each block.
        layers_per_block (`int`, *optional*, defaults to 2):
            The number of layers per block.
        double_z (`bool`, *optional*, defaults to `True`):
            Whether to double the number of output channels for the last block.
        gradient_checkpointing (`bool`, *optional*, defaults to `False`):
            Whether to switch on gradient checkpointing.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        block_out_channels: Tuple[int, ...] = (64,),
        layers_per_block: int = 2,
        double_z: bool = True,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.layers_per_block = layers_per_block

        self.conv_in = nn.Conv2d(
            in_channels,
            block_out_channels[0],
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.down_blocks = nn.ModuleList([])

        # down
        output_channel = block_out_channels[0]
        for i in range(len(block_out_channels) - 1):
            input_channel = output_channel
            output_channel = block_out_channels[i]

            down_block = DownEncoderBlock2D(
                num_layers=self.layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
            )
            self.down_blocks.append(down_block)

        # final
        self.final_block = FinalBlock2D(
            in_channels=output_channel,
            out_channels=block_out_channels[-1],
            num_layers=self.layers_per_block,
        )

        # out
        self.conv_norm_out = nn.GroupNorm(
            num_channels=block_out_channels[-1], num_groups=32, eps=1e-6
        )
        self.conv_act = nn.SiLU()

        conv_out_channels = 2 * out_channels if double_z else out_channels
        self.conv_out = nn.Conv2d(block_out_channels[-1], conv_out_channels, 3, padding=1)

        self.gradient_checkpointing = gradient_checkpointing

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        r"""The forward method of the `Encoder` class."""

        sample = self.conv_in(sample)

        if self.training and self.gradient_checkpointing:
            # down
            for down_block in self.down_blocks:
                sample = torch.utils.checkpoint.checkpoint(down_block, sample, use_reentrant=False)

            # final
            sample = torch.utils.checkpoint.checkpoint(
                self.final_block, sample, use_reentrant=False
            )

        else:
            # down
            for down_block in self.down_blocks:
                sample = down_block(sample)

            # final
            sample = self.final_block(sample)

        # post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        return sample


class UpDecoderBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
    ):
        super().__init__()
        resnets = []

        for i in range(num_layers):
            input_channels = in_channels if i == 0 else out_channels

            resnets.append(
                ResnetBlock2D(
                    in_channels=input_channels, out_channels=out_channels, dropout=dropout
                )
            )

        self.resnets = nn.ModuleList(resnets)
        # NOTE: DO NOT USE SEQUENTIAL HERE.

        self.upsampler = Upsample2D(out_channels)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states)

        hidden_states = self.upsampler(hidden_states)

        return hidden_states


class Decoder2D(nn.Module):
    r"""
    The `Decoder` layer of a variational autoencoder
        that decodes its latent representation into an output sample.

    Args:
        in_channels (`int`, *optional*, defaults to 3): The number of input channels.
        out_channels (`int`, *optional*, defaults to 3): The number of output channels.
        block_out_channels (`Tuple[int, ...]`, *optional*, defaults to `(64,)`):
                            The number of output channels for each block.
        layers_per_block (`int`, *optional*, defaults to 2): The number of layers per block.
        gradient_checkpointing (`bool`, *optional*, defaults to `False`):
            Whether to switch on gradient checkpointing.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        block_out_channels: Tuple[int, ...] = (64,),
        layers_per_block: int = 2,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.layers_per_block = layers_per_block

        self.conv_in = nn.Conv2d(
            in_channels, block_out_channels[0], kernel_size=3, stride=1, padding=1
        )

        self.up_blocks = nn.ModuleList([])

        # up
        output_channel = block_out_channels[0]
        for i in range(len(block_out_channels) - 1):
            prev_output_channel = output_channel
            output_channel = block_out_channels[i]

            up_block = UpDecoderBlock2D(
                num_layers=self.layers_per_block,
                in_channels=prev_output_channel,
                out_channels=output_channel,
            )
            self.up_blocks.append(up_block)

        # final
        self.final_block = FinalBlock2D(
            in_channels=output_channel,
            out_channels=block_out_channels[-1],
            num_layers=self.layers_per_block,
        )

        # out
        self.conv_norm_out = nn.GroupNorm(
            num_channels=block_out_channels[-1], num_groups=32, eps=1e-6
        )
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(block_out_channels[-1], out_channels, 3, padding=1)
        self.gradient_checkpointing = gradient_checkpointing

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        r"""The forward method of the `Decoder` class."""

        sample = self.conv_in(sample)

        if self.training and self.gradient_checkpointing:
            # up
            for up_block in self.up_blocks:
                sample = torch.utils.checkpoint.checkpoint(
                    up_block,
                    sample,
                    use_reentrant=False,
                )

            # final
            sample = torch.utils.checkpoint.checkpoint(
                self.final_block,
                sample,
                use_reentrant=False,
            )

        else:
            # up
            for up_block in self.up_blocks:
                sample = up_block(sample)

            # final
            sample = self.final_block(sample)

        # post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        return sample


class AutoencoderKL(nn.Module):
    r"""
    A VAE model with KL loss for encoding images into latents
    and decoding latent representations into images.
    """

    _supports_gradient_checkpointing = True
    _no_split_modules = ["ResnetBlock2D"]

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
        gradient_checkpointing: bool = False,
        spatial_downsample_factor: int = 1,
        *args,
        **kwargs,
    ):
        super().__init__()

        assert 2 ** (len(enc_block_out_channels) - 1) == spatial_downsample_factor
        assert 2 ** (len(dec_block_out_channels) - 1) == spatial_downsample_factor

        self.spatial_downsample_factor = spatial_downsample_factor

        # pass init params to Encoder
        self.encoder = Encoder2D(
            in_channels=in_channels,
            out_channels=latent_channels,
            block_out_channels=enc_block_out_channels,
            layers_per_block=enc_layers_per_block,
            double_z=True,
            gradient_checkpointing=gradient_checkpointing,
        )

        # pass init params to Decoder
        self.decoder = Decoder2D(
            in_channels=latent_channels,
            out_channels=out_channels,
            block_out_channels=dec_block_out_channels,
            layers_per_block=dec_layers_per_block,
            gradient_checkpointing=gradient_checkpointing,
        )

        self.quant_conv = (
            nn.Conv2d(2 * latent_channels, 2 * latent_channels, 1) if use_quant_conv else None
        )
        self.post_quant_conv = (
            nn.Conv2d(latent_channels, latent_channels, 1) if use_post_quant_conv else None
        )

    def forward(self, x: torch.FloatTensor) -> CausalAutoencoderOutput:
        z, p = self.encode(x)
        assert x.size(-2) // z.size(-2) == self.spatial_downsample_factor
        assert x.size(-1) // z.size(-1) == self.spatial_downsample_factor
        x = self.decode(z).sample
        return CausalAutoencoderOutput(x, z, p)

    def encode(self, x: torch.FloatTensor) -> CausalEncoderOutput:
        h = self.encoder(x)
        h = self.quant_conv(h) if self.quant_conv is not None else h
        p = DiagonalGaussianDistribution(h)
        z = p.sample()
        return CausalEncoderOutput(z, p)

    def decode(self, z: torch.FloatTensor) -> CausalDecoderOutput:
        z = self.post_quant_conv(z) if self.post_quant_conv is not None else z
        x = self.decoder(z)
        return CausalDecoderOutput(x)
