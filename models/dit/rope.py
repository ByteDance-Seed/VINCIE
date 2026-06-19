# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
from functools import lru_cache
from typing import Optional, Tuple
import torch
from einops import rearrange
from torch import broadcast_tensors, nn
from .rotary_embedding_torch import RotaryEmbedding, apply_rotary_emb

from common.cache_v2 import Cache


class RotaryEmbeddingBase(nn.Module):
    def __init__(self, dim: int, rope_dim: int):
        super().__init__()
        self.rope = RotaryEmbedding(
            dim=dim // rope_dim,
            freqs_for="pixel",
            max_freq=256,
        )
        # 1. Set model.requires_grad_(True) after model creation will make
        #    the `requires_grad=False` for rope freqs no longer hold.
        # 2. Even if we don't set requires_grad_(True) explicitly,
        #    FSDP is not memory efficient when handling fsdp_wrap
        #    with mixed requires_grad=True/False.
        # With above consideration, it is easier just remove the freqs
        # out of nn.Parameters when `learned_freq=False`
        freqs = self.rope.freqs
        del self.rope.freqs
        self.rope.register_buffer("freqs", freqs.data)

    @lru_cache(maxsize=128)
    def get_axial_freqs(self, *dims):
        return self.rope.get_axial_freqs(*dims)


class RotaryEmbedding3d(RotaryEmbeddingBase):
    def __init__(self, dim: int):
        super().__init__(dim, rope_dim=3)
        self.mm = False

    def forward(
        self,
        q: torch.FloatTensor,  # b h l d
        k: torch.FloatTensor,  # b h l d
        size: Tuple[int, int, int],
    ) -> Tuple[
        torch.FloatTensor,
        torch.FloatTensor,
    ]:
        T, H, W = size
        freqs = self.get_axial_freqs(T, H, W)
        q = rearrange(q, "b h (T H W) d -> b h T H W d", T=T, H=H, W=W)
        k = rearrange(k, "b h (T H W) d -> b h T H W d", T=T, H=H, W=W)
        q = apply_rotary_emb(freqs, q.float()).to(q.dtype)
        k = apply_rotary_emb(freqs, k.float()).to(k.dtype)
        q = rearrange(q, "b h T H W d -> b h (T H W) d")
        k = rearrange(k, "b h T H W d -> b h (T H W) d")
        return q, k


class NaRotaryEmbedding3d(RotaryEmbedding3d):
    def forward(
        self,
        q: torch.FloatTensor,  # L h d
        k: torch.FloatTensor,  # L h d
        shape: torch.LongTensor,
        cache: Cache,
    ) -> Tuple[
        torch.FloatTensor,
        torch.FloatTensor,
    ]:
        freqs = cache("rope_freqs_3d", self.get_freqs, shape)
        q = rearrange(q, "L h d -> h L d")
        k = rearrange(k, "L h d -> h L d")
        q = apply_rotary_emb(freqs, q.float()).to(q.dtype)
        k = apply_rotary_emb(freqs, k.float()).to(k.dtype)
        q = rearrange(q, "h L d -> L h d")
        k = rearrange(k, "h L d -> L h d")
        return q, k

    def get_freqs(
        self,
        shape: torch.LongTensor,
    ) -> torch.Tensor:
        freq_list = []
        for f, h, w in shape.tolist():
            freqs = self.get_axial_freqs(f, h, w)
            freq_list.append(freqs.view(-1, freqs.size(-1)))
        return torch.cat(freq_list, dim=0)


class MMRotaryEmbeddingBase(nn.Module):
    def __init__(self, dim: int, rope_dim: int):
        super().__init__()
        self.rope = RotaryEmbedding(
            dim=dim // rope_dim,
            freqs_for="lang",
            theta=10000,
        )
        freqs = self.rope.freqs
        del self.rope.freqs
        self.rope.register_buffer("freqs", freqs.data)
        self.mm = True

    def get_axial_freqs(self, *dims, offsets):
        assert self.rope.freqs_for == "lang"
        Colon = slice(None)
        all_freqs = []
        for ind, dim in enumerate(dims):
            start = offsets[ind]
            end = start + int(dim)
            pos = torch.arange(start, end, device=self.rope.device)
            freqs = self.rope.forward(pos, seq_len=dim, offset=offsets[ind])
            all_axis = [None] * len(dims)
            all_axis[ind] = Colon
            new_axis_slice = (Ellipsis, *all_axis, Colon)
            all_freqs.append(freqs[new_axis_slice])
        all_freqs = broadcast_tensors(*all_freqs)
        return torch.cat(all_freqs, dim=-1)


class NaMMRotaryEmbedding3d(MMRotaryEmbeddingBase):
    def __init__(self, dim: int):
        super().__init__(dim, rope_dim=3)

    def forward(
        self,
        vid_q: torch.FloatTensor,  # L h d
        vid_k: torch.FloatTensor,  # L h d
        vid_shape: torch.LongTensor,  # B 3
        txt_q: torch.FloatTensor,  # L h d
        txt_k: torch.FloatTensor,  # L h d
        txt_shape: torch.LongTensor,  # B 1
        cache: Cache,
    ) -> Tuple[
        torch.FloatTensor,
        torch.FloatTensor,
        torch.FloatTensor,
        torch.FloatTensor,
    ]:
        vid_freqs, txt_freqs = cache("mmrope_freqs_3d", self.get_freqs, vid_shape, txt_shape)
        vid_q = rearrange(vid_q, "L h d -> h L d")
        vid_k = rearrange(vid_k, "L h d -> h L d")
        vid_q = apply_rotary_emb(vid_freqs, vid_q.float()).to(vid_q.dtype)
        vid_k = apply_rotary_emb(vid_freqs, vid_k.float()).to(vid_k.dtype)
        vid_q = rearrange(vid_q, "h L d -> L h d")
        vid_k = rearrange(vid_k, "h L d -> L h d")

        txt_q = rearrange(txt_q, "L h d -> h L d")
        txt_k = rearrange(txt_k, "L h d -> h L d")
        txt_q = apply_rotary_emb(txt_freqs, txt_q.float()).to(txt_q.dtype)
        txt_k = apply_rotary_emb(txt_freqs, txt_k.float()).to(txt_k.dtype)
        txt_q = rearrange(txt_q, "h L d -> L h d")
        txt_k = rearrange(txt_k, "h L d -> L h d")
        return vid_q, vid_k, txt_q, txt_k

    def get_freqs(
        self,
        vid_shape: torch.LongTensor,
        txt_shape: torch.LongTensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        # Usage of `rope.get_axial_freqs` is intentional,
        # prevent caching this large tensor (7.875 GiB).
        vid_freq_list, txt_freq_list = [], []
        for (f, h, w), l in zip(vid_shape.tolist(), txt_shape[:, 0].tolist()):
            vid_freq = self.get_axial_freqs(f, h, w, offsets=[l, 0, 0])
            vid_freq = vid_freq.reshape(-1, vid_freq.size(-1))
            txt_freq = (
                self.get_axial_freqs(l, offsets=[0]).repeat(1, 3).reshape(-1, vid_freq.size(-1))
            )
            vid_freq_list.append(vid_freq)
            txt_freq_list.append(txt_freq)
        return torch.cat(vid_freq_list, dim=0), torch.cat(txt_freq_list, dim=0)


class NaMMRotaryEmbedding4d(MMRotaryEmbeddingBase):
    def __init__(self, dim: int):
        super().__init__(dim, rope_dim=4)
        self.start_index = dim // 4 * 3

    def forward(
        self,
        vid_q: torch.FloatTensor,  # L h d
        vid_k: torch.FloatTensor,  # L h d
        vid_shape: torch.LongTensor,  # B 3
        txt_q: torch.FloatTensor,  # L h d
        txt_k: torch.FloatTensor,  # L h d
        txt_shape: torch.LongTensor,  # B 1
        cache: Cache,
    ) -> Tuple[
        torch.FloatTensor,
        torch.FloatTensor,
        torch.FloatTensor,
        torch.FloatTensor,
    ]:
        vid_freqs, txt_freqs = cache("mmrope_freqs_4d", self.get_freqs, vid_shape, txt_shape)
        vid_q = rearrange(vid_q, "L h d -> h L d")
        vid_k = rearrange(vid_k, "L h d -> h L d")
        vid_q = apply_rotary_emb(vid_freqs, vid_q.float()).to(vid_q.dtype)
        vid_k = apply_rotary_emb(vid_freqs, vid_k.float()).to(vid_k.dtype)
        vid_q = rearrange(vid_q, "h L d -> L h d")
        vid_k = rearrange(vid_k, "h L d -> L h d")

        txt_q = rearrange(txt_q, "L h d -> h L d")
        txt_k = rearrange(txt_k, "L h d -> h L d")
        txt_q = apply_rotary_emb(txt_freqs, txt_q.float(), start_index=self.start_index).to(
            txt_q.dtype
        )
        txt_k = apply_rotary_emb(txt_freqs, txt_k.float(), start_index=self.start_index).to(
            txt_q.dtype
        )
        txt_q = rearrange(txt_q, "h L d -> L h d")
        txt_k = rearrange(txt_k, "h L d -> L h d")
        return vid_q, vid_k, txt_q, txt_k

    def get_freqs(
        self,
        vid_shape: torch.LongTensor,
        txt_shape: torch.LongTensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        vid_freq_list, txt_freq_list = [], []
        for (f, h, w), l in zip(vid_shape.tolist(), txt_shape[:, 0].tolist()):
            vid_freq = self.get_axial_freqs(f, h, w, offsets=[0, 0, 0])
            vid_freq = vid_freq.reshape(-1, vid_freq.size(-1))
            txt_freq = self.get_axial_freqs(l, offsets=[0])
            txt_freq = txt_freq.reshape(-1, txt_freq.size(-1))
            vid_freq_list.append(vid_freq)
            txt_freq_list.append(txt_freq)
        return torch.cat(vid_freq_list, dim=0), torch.cat(txt_freq_list, dim=0)


class NaMMRotaryEmbedding3dV2(NaMMRotaryEmbedding3d):
    def get_freqs(
        self,
        vid_shape: torch.LongTensor,
        txt_shape: torch.LongTensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        vid_freq_list, txt_freq_list = [], []
        for (f, h, w), l in zip(vid_shape.tolist(), txt_shape[:, 0].tolist()):
            vid_freq = self.get_axial_freqs(f, h, w, offsets=[l, l, l])
            vid_freq = vid_freq.reshape(-1, vid_freq.size(-1))
            txt_freq = (
                self.get_axial_freqs(l, offsets=[0]).repeat(1, 3).reshape(-1, vid_freq.size(-1))
            )
            vid_freq_list.append(vid_freq)
            txt_freq_list.append(txt_freq)
        return torch.cat(vid_freq_list, dim=0), torch.cat(txt_freq_list, dim=0)


class NaMMRotaryEmbedding3dV3(NaMMRotaryEmbedding3d):
    def get_freqs(
        self,
        vid_shape: torch.LongTensor,
        txt_shape: torch.LongTensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        vid_freq_list, txt_freq_list = [], []
        for (f, h, w), l in zip(vid_shape.tolist(), txt_shape[:, 0].tolist()):
            vid_freq = self.get_axial_freqs(f, h, w, offsets=[0, 0, 0])
            vid_freq = vid_freq.reshape(-1, vid_freq.size(-1))
            txt_freq = (
                self.get_axial_freqs(l, offsets=[0]).repeat(1, 3).reshape(-1, vid_freq.size(-1))
            )
            vid_freq_list.append(vid_freq)
            txt_freq_list.append(txt_freq)
        return torch.cat(vid_freq_list, dim=0), torch.cat(txt_freq_list, dim=0)


def get_rope(rope_type: Optional[str], dim: int):
    if rope_type is None:
        return None
    if rope_type == "rope3d":
        return RotaryEmbedding3d(dim=dim)
    raise NotImplementedError(f"{rope_type} is not supported.")


def get_na_rope(rope_type: Optional[str], dim: int):
    if rope_type is None:
        return None
    if rope_type == "rope3d":
        return NaRotaryEmbedding3d(dim=dim)
    if rope_type == "mmrope3d":
        return NaMMRotaryEmbedding3d(dim=dim)
    if rope_type == "mmrope3d_v2":
        return NaMMRotaryEmbedding3dV2(dim=dim)
    if rope_type == "mmrope3d_v3":
        return NaMMRotaryEmbedding3dV3(dim=dim)
    if rope_type == "mmrope4d":
        return NaMMRotaryEmbedding4d(dim=dim)
    raise NotImplementedError(f"{rope_type} is not supported.")
