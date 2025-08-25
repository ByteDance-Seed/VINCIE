# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates  
# SPDX-License-Identifier: Apache-2.0
from typing import Tuple, Union
import torch
from einops import rearrange
from torch import nn
from torch.nn.modules.utils import _triple

from common.cache_v2 import Cache
from common.distributed.ops import gather_outputs, slice_inputs

from .. import na


class PatchIn(nn.Module):
    def __init__(
        self,
        in_channels: int,
        patch_size: Union[int, Tuple[int, int, int]],
        dim: int,
    ):
        super().__init__()
        t, h, w = _triple(patch_size)
        self.patch_size = t, h, w
        self.img_proj = nn.Linear(
            in_channels * 1 * h * w, dim
        )  # img_proj is used for image or 1st frame
        if t > 1:
            self.vid_proj = nn.Linear(
                in_channels * t * h * w, dim
            )  # if t==1, self.vid_proj is deprecated; only self.img_proj is used

    def forward(
        self,
        vid: torch.Tensor,
    ) -> torch.Tensor:
        t, h, w = self.patch_size
        if t > 1:
            assert vid.size(2) % t == 1
            img = rearrange(
                vid[:, :, :1], "b c (T t) (H h) (W w) -> b T H W (t h w c)", t=1, h=h, w=w
            )
            img = self.img_proj(img)
            if vid.size(2) > 1:
                clip = rearrange(
                    vid[:, :, 1:], "b c (T t) (H h) (W w) -> b T H W (t h w c)", t=t, h=h, w=w
                )
                clip = self.vid_proj(clip)
                img = torch.cat([img, clip], dim=1)
            vid = img
        else:
            vid = rearrange(vid, "b c (T t) (H h) (W w) -> b T H W (t h w c)", t=t, h=h, w=w)
            vid = self.img_proj(vid)
        return vid


class PatchOut(nn.Module):
    def __init__(
        self,
        out_channels: int,
        patch_size: Union[int, Tuple[int, int, int]],
        dim: int,
    ):
        super().__init__()
        t, h, w = _triple(patch_size)
        self.patch_size = t, h, w
        self.img_proj = nn.Linear(dim, out_channels * 1 * h * w)
        if t > 1:
            self.vid_proj = nn.Linear(dim, out_channels * t * h * w)

    def forward(
        self,
        vid: torch.Tensor,
    ) -> torch.Tensor:
        t, h, w = self.patch_size
        if t > 1:
            img = self.img_proj(vid[:, :, :1])
            img = rearrange(img, "b T H W (t h w c) -> b c (T t) (H h) (W w)", t=1, h=h, w=w)
            if vid.size(2) > 1:
                clip = self.vid_proj(vid[:, :, 1:])
                clip = rearrange(clip, "b T H W (t h w c) -> b c (T t) (H h) (W w)", t=t, h=h, w=w)
                img = torch.cat([img, clip], dim=2)
        else:
            vid = self.img_proj(vid)
            vid = rearrange(vid, "b T H W (t h w c) -> b c (T t) (H h) (W w)", t=t, h=h, w=w)

        return vid


class NaPatchIn(PatchIn):
    def forward(
        self,
        vid: torch.Tensor,  # l c
        vid_shape: torch.LongTensor,
        cache: Cache = Cache(disable=True),  # for test
    ) -> torch.Tensor:
        t, h, w = self.patch_size
        if not (t == h == w == 1):
            vid = na.unflatten(vid, vid_shape)
            for i in range(len(vid)):
                cur_vid = vid[i]
                if t > 1:
                    assert vid[i].size(0) % t == 1
                img = rearrange(
                    cur_vid[:1], "(T t) (H h) (W w) c -> T H W (t h w c)", t=1, h=h, w=w
                )
                img = self.img_proj(img)
                if cur_vid.size(0) > 1:
                    clip = rearrange(
                        cur_vid[1:], "(T t) (H h) (W w) c -> T H W (t h w c)", t=t, h=h, w=w
                    )
                    clip = self.vid_proj(clip)
                    img = torch.cat([img, clip], dim=0)
                vid[i] = img
            vid, vid_shape = na.flatten(vid)
        else:
            # when t == h == w ==1, no rearrangement is needed
            vid = self.img_proj(vid)

        # slice vid after patching in when using sequence parallelism
        vid = slice_inputs(vid, dim=0)

        return vid, vid_shape


class NaPatchOut(PatchOut):
    def forward(
        self,
        vid: torch.FloatTensor,  # l c
        vid_shape: torch.LongTensor,
        cache: Cache = Cache(disable=True),  # for test
    ) -> Tuple[
        torch.FloatTensor,
        torch.LongTensor,
    ]:
        t, h, w = self.patch_size
        # gather vid before patching out when enabling sequence parallelism
        vid = gather_outputs(
            vid,
            gather_dim=0,
            padding_dim=0,
            unpad_shape=vid_shape,
        )
        if not (t == h == w == 1):
            vid = na.unflatten(vid, vid_shape)
            for i in range(len(vid)):
                cur_vid = vid[i]
                if t > 1:
                    img = self.img_proj(cur_vid[:1])
                    img = rearrange(img, "T H W (t h w c) -> (T t) (H h) (W w) c", t=1, h=h, w=w)
                    if cur_vid.size(0) > 1:
                        clip = self.vid_proj(cur_vid[1:])
                        clip = rearrange(
                            clip, "T H W (t h w c) -> (T t) (H h) (W w) c", t=t, h=h, w=w
                        )
                        img = torch.cat([img, clip], dim=0)
                    vid[i] = img
                else:
                    cur_vid = self.img_proj(cur_vid)
                    vid[i] = rearrange(
                        cur_vid, "T H W (t h w c) -> (T t) (H h) (W w) c", t=t, h=h, w=w
                    )
            vid, vid_shape = na.flatten(vid)
        else:
            vid = self.img_proj(vid)

        return vid, vid_shape
