# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates  
# SPDX-License-Identifier: Apache-2.0
import importlib
from functools import partial
from typing import Callable, List, Optional
import torch
from einops import rearrange
from torch import nn

from common.cache_v2 import Cache
from common.distributed.advanced import (
    get_sequence_parallel_group,
    get_sequence_parallel_world_size,
)
from common.distributed.ops import slice_inputs

# (dim: int, emb_dim: int)
ada_layer_type = Callable[[int, int], nn.Module]
turbox = None


def get_ada_layer(ada_layer: str, norm_type: Optional[str]) -> ada_layer_type:
    if ada_layer == "single":
        return AdaSingle
    if ada_layer == "fused_single" and norm_type == "fusedrms":
        global turbox
        if turbox is None:
            turbox = importlib.import_module("turbox")
        return FusedAdaSingleRMSNorm
    raise NotImplementedError(f"{ada_layer} is not supported")


def expand_dims(x: torch.Tensor, dim: int, ndim: int):
    """
    Expand tensor "x" to "ndim" by adding empty dims at "dim".
    Example: x is (b d), target ndim is 5, add dim at 1, return (b 1 1 1 d).
    """
    shape = x.shape
    shape = shape[:dim] + (1,) * (ndim - len(shape)) + shape[dim:]
    return x.reshape(shape)


class AdaSingle(nn.Module):
    def __init__(
        self,
        dim: int,
        emb_dim: int,
        layers: List[str],
        modes: List[str] = ["in", "out"],
    ):
        super().__init__()
        self.dim = dim
        self.emb_dim = emb_dim
        self.layers = layers
        for l in layers:
            if "in" in modes:
                self.register_parameter(f"{l}_shift", nn.Parameter(torch.randn(dim) / dim**0.5))
                self.register_parameter(
                    f"{l}_scale", nn.Parameter(torch.randn(dim) / dim**0.5 + 1)
                )
            if "out" in modes:
                self.register_parameter(f"{l}_gate", nn.Parameter(torch.randn(dim) / dim**0.5))

    def forward(
        self,
        hid: torch.FloatTensor,  # b ... c
        emb: torch.FloatTensor,  # b d
        layer: str,
        mode: str,
        cache: Cache = Cache(disable=True),
        branch_tag: str = "",
        hid_shape: Optional[torch.LongTensor] = None,  # b
        norm_layer: Optional[torch.nn.Module] = None,
        residual: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        idx = self.layers.index(layer)
        emb = cache(f"emb_{idx}", lambda: self.expand_rearrange_emb(hid, emb, hid_shape, idx))

        if hid_shape is not None:
            seq_idx, sliced_hid_len, sliced_vid_bsz, sliced_vid_len, sliced_total_vid_len = cache(
                f"sliced_hid_len_{branch_tag}",
                partial(self.get_sliced_hid_len, branch_tag=branch_tag),
                hid_shape,
            )

            # Memory-performance trade-off
            # - for loop -> high cpu overhead & low cache memory (bsz * emb_dim)
            # - repeat -> low cpu overhead & high cache memory (sliced_hid_len.sum() * emb_dim)
            emb = cache(
                f"emb_repeat_{idx}_{branch_tag}",
                lambda: torch.cat(
                    [
                        (
                            emb[bid : bid + 1]
                            if i < sliced_vid_bsz
                            else emb[bid].repeat(slen, *([1] * (emb.ndim - 1)))
                        )
                        for i, (bid, slen) in enumerate(zip(seq_idx, sliced_hid_len))
                    ]
                ),
            )

        shiftA, scaleA, gateA = emb.unbind(-1)
        shiftB, scaleB, gateB = (
            getattr(self, f"{layer}_shift", None),
            getattr(self, f"{layer}_scale", None),
            getattr(self, f"{layer}_gate", None),
        )

        if mode == "in":
            hid = norm_layer(hid)
            if hid_shape is not None:
                out = [
                    h * (scaleA[i] + scaleB) + (shiftA[i] + shiftB)
                    for i, h in enumerate(hid[:sliced_total_vid_len].split(sliced_vid_len))
                ]
                if sliced_total_vid_len < hid.size(0):
                    out.append(
                        hid[sliced_total_vid_len:] * (scaleA[sliced_vid_bsz:] + scaleB)
                        + (shiftA[sliced_vid_bsz:] + shiftB)
                    )
                return torch.cat(out)
            return hid.mul_(scaleA + scaleB).add_(shiftA + shiftB)
        if mode == "out":
            if hid_shape is not None:
                out = [
                    h * (gateA[i] + gateB)
                    for i, h in enumerate(hid[:sliced_total_vid_len].split(sliced_vid_len))
                ]
                if sliced_total_vid_len < hid.size(0):
                    out.append(hid[sliced_total_vid_len:] * (gateA[sliced_vid_bsz:] + gateB))
                return torch.cat(out) + residual
            return hid.mul_(gateA + gateB) + residual
        raise NotImplementedError

    def expand_rearrange_emb(self, hid, emb, hid_shape, idx):
        emb = rearrange(emb, "b (d l g) -> b d l g", l=len(self.layers), g=3)[..., idx, :]
        emb = expand_dims(emb, 1, hid.ndim + 1)
        if hid_shape is not None and get_sequence_parallel_group() is not None:
            emb = nn.functional.pad(emb, (0, 0, 0, 0, 0, 1))
        return emb

    @staticmethod
    def get_sliced_hid_len(hid_shape: torch.LongTensor, branch_tag: str):
        hid_len = hid_shape.prod(-1)
        if branch_tag == "vid":
            video_batch_size = (hid_shape[:, 0] > 1).sum().item()
        else:
            video_batch_size = 0

        sp_world = get_sequence_parallel_world_size()
        dim_size = hid_len.sum().item()

        # Get the seqlen of each sample this sp_rank.
        batch_size = hid_len.size(0)
        seq_idx = torch.arange(batch_size, device=hid_len.device).repeat_interleave(hid_len)
        if dim_size % sp_world:
            padding_size = sp_world - (dim_size % sp_world)
            seq_idx = nn.functional.pad(seq_idx, (0, padding_size), value=batch_size)
        seq_idx = slice_inputs(seq_idx, 0)
        seq_idx, sliced_hid_len = torch.unique(seq_idx, return_counts=True)

        sliced_video_batch_size = (seq_idx < video_batch_size).sum().item()
        seq_idx = seq_idx.tolist()
        sliced_hid_len = sliced_hid_len.tolist()
        sliced_video_hid_len = sliced_hid_len[:sliced_video_batch_size]
        sliced_video_total_hid_len = sum(sliced_video_hid_len)
        return (
            seq_idx,
            sliced_hid_len,
            sliced_video_batch_size,
            sliced_video_hid_len,
            sliced_video_total_hid_len,
        )


class IndexSelectDim0Func(torch.autograd.Function):
    """torch backward for x[index] is not compatible with gradient_checkpointing + cache"""

    @staticmethod
    def forward(ctx, x: torch.Tensor, index: torch.Tensor):
        ctx.index = index
        ctx.shape = tuple(x.shape)
        return x[index]

    @staticmethod
    def backward(ctx, grad_output):
        out = torch.zeros(ctx.shape, device=grad_output.device, dtype=grad_output.dtype)
        out[ctx.index] = grad_output
        return out, None, None


class FusedAdaSingleRMSNorm(AdaSingle):
    def forward(
        self,
        hid: torch.FloatTensor,  # b ... c
        emb: torch.FloatTensor,  # b d
        layer: str,
        mode: str,
        cache: Cache = Cache(disable=True),
        branch_tag: str = "",
        hid_shape: Optional[torch.LongTensor] = None,  # b
        norm_layer: Optional[torch.nn.Module] = None,
        residual: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        idx = self.layers.index(layer)
        emb = cache(f"emb_{idx}", lambda: self.expand_rearrange_emb(hid, emb, hid_shape, idx))

        if hid_shape is not None:
            seq_idx, sliced_hid_len = cache(
                f"sliced_hid_len_{branch_tag}", self.get_sliced_hid_len, hid_shape
            )
            emb = cache(
                f"emb_{idx}_{branch_tag}",
                lambda: IndexSelectDim0Func.apply(emb, seq_idx),
            )

        shiftA, scaleA, gateA = cache(
            f"unbind_contiguous_emb_{idx}_{branch_tag}",
            lambda: list(map(lambda x: x.contiguous(), emb.unbind(-1))),
        )

        if mode == "in":
            shiftB, scaleB = (
                getattr(self, f"{layer}_shift", None),
                getattr(self, f"{layer}_scale", None),
            )
            if hid_shape is not None:
                return turbox.fuse_rms_ada_func(
                    input=hid,
                    gamma=norm_layer.weight,
                    scaleA=scaleA,
                    shiftA=shiftA,
                    scaleB=scaleB,
                    shiftB=shiftB,
                    lens=sliced_hid_len,
                    eps=norm_layer.eps,
                )
            return norm_layer(hid).mul_(scaleA + scaleB).add_(shiftA + shiftB)

        if mode == "out":
            gateB = getattr(self, f"{layer}_gate", None)
            if hid_shape is not None:
                return turbox.fuse_ada_res_func(
                    input=hid,
                    gateA=gateA,
                    gateB=gateB,
                    residual=residual,
                    lens=sliced_hid_len,
                )
            return hid.mul_(gateA + gateB) + residual

        raise NotImplementedError

    @staticmethod
    def get_sliced_hid_len(hid_shape: torch.LongTensor):
        hid_len = hid_shape.prod(-1)
        sp_world = get_sequence_parallel_world_size()
        dim_size = hid_len.sum().item()

        # Get the seqlen of each sample this sp_rank.
        batch_size = hid_len.size(0)
        seq_idx = torch.arange(batch_size, device=hid_len.device).repeat_interleave(hid_len)
        if dim_size % sp_world:
            padding_size = sp_world - (dim_size % sp_world)
            seq_idx = nn.functional.pad(seq_idx, (0, padding_size), value=batch_size)
        seq_idx = slice_inputs(seq_idx, 0)
        seq_idx, sliced_hid_len = torch.unique(seq_idx, return_counts=True)

        return seq_idx, sliced_hid_len.int()
