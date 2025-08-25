# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates  
# SPDX-License-Identifier: Apache-2.0
from typing import Dict
import torch
from einops import rearrange
from torch import nn


class PatchDiscriminator2d(nn.Sequential):
    """
    PatchGAN image discriminator as used in stable diffusion VAE.
    """

    def __init__(self, in_channels: int = 3):
        super().__init__(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1),
        )
        self.reset_parameters()

    def reset_parameters(self):
        for module in self:
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, 0.0, 0.02)
            if isinstance(module, nn.BatchNorm2d):
                nn.init.normal_(module.weight, 1.0, 0.02)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            return super().forward(x)
        if x.ndim == 5:
            f = x.shape[2]
            x = rearrange(x, "b c f h w -> (b f) c h w")
            x = super().forward(x)
            x = rearrange(x, "(b f) c h w -> b c f h w", f=f)
            return x
        raise NotImplementedError


class PatchDiscriminator3d(nn.Sequential):
    """
    PatchGAN video discriminator as modified from the image discriminator.
    """

    def __init__(self, in_channels: int = 3):
        super().__init__(
            nn.Conv3d(in_channels, 64, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(64, 128, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=1, bias=False),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(128, 256, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=1, bias=False),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(256, 512, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=1, bias=False),
            nn.BatchNorm3d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(512, 1, kernel_size=4, stride=1, padding=1),
        )
        self.reset_parameters()

    def reset_parameters(self):
        for module in self:
            if isinstance(module, nn.Conv3d):
                nn.init.normal_(module.weight, 0.0, 0.02)
            if isinstance(module, nn.BatchNorm3d):
                nn.init.normal_(module.weight, 1.0, 0.02)
                nn.init.zeros_(module.bias)


class PatchDiscriminatorJoint(nn.Module):
    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.img_disc = PatchDiscriminator2d(in_channels)
        self.vid_disc = PatchDiscriminator3d(in_channels)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        v_score = self.vid_disc(x)
        i_score = self.img_disc(x)
        return dict(i_score=i_score, v_score=v_score)


if __name__ == "__main__":
    model_2d = PatchDiscriminator2d()
    model_2d_size = sum(p.numel() for p in list(model_2d.parameters()) if p.requires_grad)
    model_3d = PatchDiscriminator3d()
    model_3d_size = sum(p.numel() for p in list(model_3d.parameters()) if p.requires_grad)
    print(model_2d)
    print(model_3d)
    print(f"2D Discriminator (M) - {round(model_2d_size / 1e6, 3)}")
    print(f"3D Discriminator (M) - {round(model_3d_size / 1e6, 3)}")
