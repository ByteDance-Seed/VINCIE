from typing import Dict, Literal, Sequence
import torch
from einops import rearrange
from torch import nn
from torch.nn import BatchNorm2d, BatchNorm3d, Conv2d, Conv3d, GroupNorm, LeakyReLU, PixelUnshuffle
from torch.nn.utils import spectral_norm

from models.ae.modules.causal_inflation_lib import causal_class_wrapper
from models.ae.modules.types import _norm_type_t


def get_conv_wrapper(norm_type: _norm_type_t, module: nn.Module):
    if norm_type == "SpectralNorm":
        return spectral_norm(module)
    else:
        return module


def get_norm_layer(norm_type: _norm_type_t, channel: int, ndim: int):
    if norm_type == "BatchNorm":
        if ndim == 2:
            return BatchNorm2d(
                num_features=channel,
                eps=1e-05,
                momentum=0.1,
                affine=True,
                track_running_stats=True,
            )
        elif ndim == 3:
            return BatchNorm3d(
                num_features=channel,
                eps=1e-05,
                momentum=0.1,
                affine=True,
                track_running_stats=True,
            )
        else:
            raise NotImplementedError
    elif norm_type == "GroupNorm":
        return GroupNorm(num_groups=32, num_channels=channel)
    elif norm_type == "SpectralNorm":
        return nn.Identity()
    raise NotImplementedError


class PatchDiscriminator2d(nn.Module):
    """
    PatchGAN image discriminator as used in stable diffusion VAE.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        channel_mult: Sequence[int] = (1, 2, 4, 8),
        norm_type: _norm_type_t = "BatchNorm",
        patch_size: int = 1,
    ):
        super().__init__()
        self.norm_type = norm_type
        self.patch_in = PixelUnshuffle(downscale_factor=patch_size)
        self.discriminator = nn.Sequential(
            get_conv_wrapper(
                norm_type,
                Conv2d(
                    in_channels=in_channels * (patch_size**2),
                    out_channels=base_channels * channel_mult[0],
                    kernel_size=4,
                    stride=2,
                    padding=1,
                ),
            ),
            LeakyReLU(negative_slope=0.2, inplace=True),
            get_conv_wrapper(
                norm_type,
                Conv2d(
                    in_channels=base_channels * channel_mult[0],
                    out_channels=base_channels * channel_mult[1],
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=(norm_type != "BatchNorm"),
                ),
            ),
            get_norm_layer(norm_type, channel=base_channels * channel_mult[1], ndim=2),
            LeakyReLU(negative_slope=0.2, inplace=True),
            get_conv_wrapper(
                norm_type,
                Conv2d(
                    in_channels=base_channels * channel_mult[1],
                    out_channels=base_channels * channel_mult[2],
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=(norm_type != "BatchNorm"),
                ),
            ),
            get_norm_layer(norm_type, channel=base_channels * channel_mult[2], ndim=2),
            LeakyReLU(negative_slope=0.2, inplace=True),
            get_conv_wrapper(
                norm_type,
                Conv2d(
                    in_channels=base_channels * channel_mult[2],
                    out_channels=base_channels * channel_mult[3],
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=(norm_type != "BatchNorm"),
                ),
            ),
            get_norm_layer(norm_type, channel=base_channels * channel_mult[3], ndim=2),
            LeakyReLU(negative_slope=0.2, inplace=True),
            get_conv_wrapper(
                norm_type,
                Conv2d(
                    in_channels=base_channels * channel_mult[3],
                    out_channels=1,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
            ),
        )
        self.reset_parameters()

    def reset_parameters(self):
        for module in self.discriminator:
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, 0.0, 0.02)
            if isinstance(module, nn.BatchNorm2d):
                nn.init.normal_(module.weight, 1.0, 0.02)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert self.norm_type != "SpectralNorm" or hasattr(self.discriminator[0], "weight_u")
        if x.ndim == 4:
            return self.discriminator(self.patch_in(x))
        if x.ndim == 5:
            f = x.shape[2]
            x = rearrange(x, "b c f h w -> (b f) c h w")
            x = self.discriminator(self.patch_in(x))
            x = rearrange(x, "(b f) c h w -> b c f h w", f=f)
            return x
        raise NotImplementedError


class PatchDiscriminator3d(nn.Sequential):
    """
    PatchGAN video discriminator as modified from the image discriminator.
    """

    def __init__(
        self,
        in_channels: int = 3,
        norm_type: _norm_type_t = "BatchNorm",
        patch_size: int = 1,
        base_channels: int = 64,
        channel_mult: Sequence[int] = (1, 2, 4, 8),
    ):
        super().__init__()
        self.norm_type = norm_type
        self.patch_in = causal_class_wrapper(PixelUnshuffle(downscale_factor=patch_size))
        self.discriminator = nn.Sequential(
            get_conv_wrapper(
                norm_type,
                Conv3d(
                    in_channels=in_channels * (patch_size**2),
                    out_channels=base_channels * channel_mult[0],
                    kernel_size=(3, 4, 4),
                    stride=(1, 2, 2),
                    padding=1,
                ),
            ),
            LeakyReLU(negative_slope=0.2, inplace=True),
            get_conv_wrapper(
                norm_type,
                Conv3d(
                    in_channels=base_channels * channel_mult[0],
                    out_channels=base_channels * channel_mult[1],
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=(norm_type != "BatchNorm"),
                ),
            ),
            get_norm_layer(norm_type, channel=base_channels * channel_mult[1], ndim=3),
            LeakyReLU(negative_slope=0.2, inplace=True),
            get_conv_wrapper(
                norm_type,
                Conv3d(
                    in_channels=base_channels * channel_mult[1],
                    out_channels=base_channels * channel_mult[2],
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=(norm_type != "BatchNorm"),
                ),
            ),
            get_norm_layer(norm_type, channel=base_channels * channel_mult[2], ndim=3),
            LeakyReLU(negative_slope=0.2, inplace=True),
            get_conv_wrapper(
                norm_type,
                Conv3d(
                    in_channels=base_channels * channel_mult[2],
                    out_channels=base_channels * channel_mult[3],
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=(norm_type != "BatchNorm"),
                ),
            ),
            get_norm_layer(norm_type, channel=base_channels * channel_mult[3], ndim=3),
            LeakyReLU(negative_slope=0.2, inplace=True),
            get_conv_wrapper(
                norm_type,
                Conv3d(
                    in_channels=base_channels * channel_mult[3],
                    out_channels=1,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
            ),
        )
        self.reset_parameters()

    def reset_parameters(self):
        for module in self.discriminator:
            if isinstance(module, nn.Conv3d):
                nn.init.normal_(module.weight, 0.0, 0.02)
            if isinstance(module, nn.BatchNorm3d):
                nn.init.normal_(module.weight, 1.0, 0.02)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert self.norm_type != "SpectralNorm" or hasattr(self.discriminator[0], "weight_u")
        return self.discriminator(self.patch_in(x))


class PatchDiscriminatorJoint(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        patch_size: int = 1,
        base_channels: int = 64,
        channel_mult: Sequence[int] = (1, 2, 4, 8),
        norm_type: _norm_type_t = "BatchNorm",
        mode: Literal["image", "video", "all"] = "all",
    ):
        super().__init__()
        self.img_disc = (
            PatchDiscriminator2d(
                in_channels=in_channels,
                base_channels=base_channels,
                channel_mult=channel_mult,
                norm_type=norm_type,
                patch_size=patch_size,
            )
            if mode != "video"
            else None
        )
        self.vid_disc = (
            PatchDiscriminator3d(
                in_channels=in_channels,
                base_channels=base_channels,
                channel_mult=channel_mult,
                norm_type=norm_type,
                patch_size=patch_size,
            )
            if mode != "image"
            else None
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Input: x: [B, C, T, H, W]
        """
        v_score = self.vid_disc(x) if self.vid_disc is not None else None
        i_score = self.img_disc(x) if self.img_disc is not None else None
        return dict(i_score=i_score, v_score=v_score)


if __name__ == "__main__":
    model_2d = PatchDiscriminator2d(base_channels=128, channel_mult=[1, 2, 2, 4])
    model_2d_size = sum(p.numel() for p in list(model_2d.parameters()) if p.requires_grad)
    x_3d = torch.randn(1, 3, 17, 128, 128)
    x_2d = torch.randn(1, 3, 128, 128)
    model_3d = PatchDiscriminator3d(base_channels=128, channel_mult=[1, 2, 2, 4])
    print(model_3d(x_3d).size())
    print(model_2d(x_2d).size())
    model_3d_size = sum(p.numel() for p in list(model_3d.parameters()) if p.requires_grad)
    print(model_2d)
    print(model_3d)
    print(f"2D Discriminator (M) - {round(model_2d_size / 1e6, 3)}")
    print(f"3D Discriminator (M) - {round(model_3d_size / 1e6, 3)}")
