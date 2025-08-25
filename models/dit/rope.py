# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates  
# SPDX-License-Identifier: Apache-2.0
from functools import lru_cache
from typing import Optional, Tuple
import torch
from einops import rearrange
from .rotary_embedding_torch import RotaryEmbedding, apply_rotary_emb
from torch import nn

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


class MMRotaryEmbeddingBase(RotaryEmbeddingBase):
    def __init__(self, dim: int, rope_dim: int):
        super().__init__(dim, rope_dim)
        self.rope = RotaryEmbedding(
            dim=dim // rope_dim,
            freqs_for="lang",
            theta=10000,
        )
        freqs = self.rope.freqs
        del self.rope.freqs
        self.rope.register_buffer("freqs", freqs.data)
        self.mm = True

class NaMMRotaryEmbedding3d(MMRotaryEmbeddingBase):
    def __init__(self, dim: int):
        super().__init__(dim, rope_dim=3)
        self.max_len = 1024

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
        assert vid_freqs.shape[0] == vid_q.shape[0], f"vid_freqs and vid_q should have the same batch size, but got {vid_freqs.shape[0]} and {vid_q.shape[0]}" 
        assert txt_freqs.shape[0] == txt_q.shape[0], f"txt_freqs and txt_q should have the same batch size, but got {txt_freqs.shape[0]} and {txt_q.shape[0]}" 
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
        max_len, max_h, max_w = 0, 0, 0
        for (f, h, w), l in zip(vid_shape.tolist(), txt_shape[:, 0].tolist()):
            max_len = l + f if l + f > max_len else max_len
            max_h = h if h > max_h else max_h
            max_w = w if w > max_w else max_w

        # Usage of `rope.get_axial_freqs` is intentional,
        vid_freqs = self.rope.get_axial_freqs(max_len, max_h, max_w)
        txt_freqs = self.get_axial_freqs(max_len)
        vid_freq_list, txt_freq_list = [], []
        for (f, h, w), l in zip(vid_shape.tolist(), txt_shape[:, 0].tolist()):
            vid_freq = vid_freqs[l : l + f, :h, :w].reshape(-1, vid_freqs.size(-1))
            txt_freq = txt_freqs[:l].repeat(1, 3).reshape(-1, vid_freqs.size(-1))
            vid_freq_list.append(vid_freq)
            txt_freq_list.append(txt_freq)
        return torch.cat(vid_freq_list, dim=0), torch.cat(txt_freq_list, dim=0)


def get_na_rope(rope_type: Optional[str], dim: int):
    if rope_type is None:
        return None
    if rope_type == "mmrope3d":
        return NaMMRotaryEmbedding3d(dim=dim)
    raise NotImplementedError(f"{rope_type} is not supported.")
