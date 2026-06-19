# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
from functools import partial
from typing import Optional, Tuple, Union
import torch
from einops import rearrange
from torch import nn
from torch.nn.modules.utils import _triple

from common.cache_v2 import Cache
from common.distributed.ops import gather_heads_scatter_seq, gather_seq_scatter_heads_qkv

from ... import na
from ...attention import FlashAttentionVarlen
from ...mm import MMArg, MMModule
from ...normalization import norm_layer_type
from ...rope import get_na_rope
from ...window import get_window_op


class NaMMAttention(nn.Module):
    def __init__(
        self,
        vid_dim: int,
        txt_dim: int,
        heads: int,
        head_dim: int,
        qk_bias: bool,
        qk_norm: norm_layer_type,
        qk_norm_eps: float,
        rope_type: Optional[str],
        rope_dim: int,
        shared_weights: bool,
        **kwargs,
    ):
        super().__init__()
        dim = MMArg(vid_dim, txt_dim)
        inner_dim = heads * head_dim
        qkv_dim = inner_dim * 3
        self.head_dim = head_dim
        self.proj_qkv = MMModule(
            nn.Linear, dim, qkv_dim, bias=qk_bias, shared_weights=shared_weights
        )
        self.proj_out = MMModule(nn.Linear, inner_dim, dim, shared_weights=shared_weights)
        self.norm_q = MMModule(
            qk_norm,
            dim=head_dim,
            eps=qk_norm_eps,
            elementwise_affine=True,
            shared_weights=shared_weights,
        )
        self.norm_k = MMModule(
            qk_norm,
            dim=head_dim,
            eps=qk_norm_eps,
            elementwise_affine=True,
            shared_weights=shared_weights,
        )

        self.rope = get_na_rope(rope_type=rope_type, dim=rope_dim)
        self.attn = FlashAttentionVarlen()

    def forward(
        self,
        vid: torch.FloatTensor,  # l c
        txt: torch.FloatTensor,  # l c
        vid_shape: torch.LongTensor,  # b 3
        txt_shape: torch.LongTensor,  # b 1
        cache: Cache,
    ) -> Tuple[
        torch.FloatTensor,
        torch.FloatTensor,
    ]:
        vid_qkv, txt_qkv = self.proj_qkv(vid, txt)
        vid_qkv = gather_seq_scatter_heads_qkv(
            vid_qkv,
            seq_dim=0,
            qkv_shape=vid_shape,
            cache=cache.namespace("vid"),
        )
        txt_qkv = gather_seq_scatter_heads_qkv(
            txt_qkv,
            seq_dim=0,
            qkv_shape=txt_shape,
            cache=cache.namespace("txt"),
        )
        vid_qkv = rearrange(vid_qkv, "l (o h d) -> l o h d", o=3, d=self.head_dim)
        txt_qkv = rearrange(txt_qkv, "l (o h d) -> l o h d", o=3, d=self.head_dim)

        vid_q, vid_k, vid_v = vid_qkv.unbind(1)
        txt_q, txt_k, txt_v = txt_qkv.unbind(1)

        vid_q, txt_q = self.norm_q(vid_q, txt_q)
        vid_k, txt_k = self.norm_k(vid_k, txt_k)

        if self.rope:
            if self.rope.mm:
                vid_q, vid_k, txt_q, txt_k = self.rope(
                    vid_q, vid_k, vid_shape, txt_q, txt_k, txt_shape, cache
                )
            else:
                vid_q, vid_k = self.rope(vid_q, vid_k, vid_shape, cache)

        vid_len = cache("vid_len", na.get_seqlen, vid_shape)
        txt_len = cache("txt_len", na.get_seqlen, txt_shape)
        all_len = cache("all_len", torch.add, vid_len, txt_len)

        concat, unconcat = cache("mm_pnp", na.concat_idx, vid_len, txt_len)

        attn = self.attn(
            q=concat(vid_q, txt_q).bfloat16(),
            k=concat(vid_k, txt_k).bfloat16(),
            v=concat(vid_v, txt_v).bfloat16(),
            cu_seqlens_q=cache("mm_seqlens", na.cu_seqlens, all_len),
            cu_seqlens_k=cache("mm_seqlens", na.cu_seqlens, all_len),
            max_seqlen_q=cache("mm_maxlen", na.max_seqlen, all_len),
            max_seqlen_k=cache("mm_maxlen", na.max_seqlen, all_len),
        ).type_as(vid_q)

        attn = rearrange(attn, "l h d -> l (h d)")
        vid_out, txt_out = unconcat(attn)
        vid_out = gather_heads_scatter_seq(vid_out, head_dim=1, seq_dim=0)
        txt_out = gather_heads_scatter_seq(txt_out, head_dim=1, seq_dim=0)

        vid_out, txt_out = self.proj_out(vid_out, txt_out)
        return vid_out, txt_out


class NaMMSpaceAttention(NaMMAttention):
    def forward(
        self,
        vid: torch.FloatTensor,  # l c
        txt: torch.FloatTensor,  # l c
        vid_shape: torch.LongTensor,  # b 3
        txt_shape: torch.LongTensor,  # b 1
        cache: Cache,
    ) -> Tuple[
        torch.FloatTensor,
        torch.FloatTensor,
    ]:
        vid_qkv, txt_qkv = self.proj_qkv(vid, txt)
        vid_qkv = gather_seq_scatter_heads_qkv(
            vid_qkv,
            seq_dim=0,
            qkv_shape=vid_shape,
            cache=cache.namespace("vid"),
        )
        txt_qkv = gather_seq_scatter_heads_qkv(
            txt_qkv,
            seq_dim=0,
            qkv_shape=txt_shape,
            cache=cache.namespace("txt"),
        )

        vid_qkv = rearrange(vid_qkv, "l (o h d) -> l o h d", o=3, d=self.head_dim)
        txt_qkv = rearrange(txt_qkv, "l (o h d) -> l o h d", o=3, d=self.head_dim)

        vid_q, vid_k, vid_v = vid_qkv.unbind(1)
        txt_q, txt_k, txt_v = txt_qkv.unbind(1)

        vid_q, txt_q = self.norm_q(vid_q, txt_q)
        vid_k, txt_k = self.norm_k(vid_k, txt_k)

        if self.rope:
            if self.rope.mm:
                vid_q, vid_k, txt_q, txt_k = self.rope(
                    vid_q, vid_k, vid_shape, txt_q, txt_k, txt_shape, cache
                )
            else:
                vid_q, vid_k = self.rope(vid_q, vid_k, vid_shape, cache)

        vid_len = cache("vid_len", na.get_seqlen, vid_shape)
        txt_len = cache("txt_len", na.get_seqlen, txt_shape)
        all_len = cache("all_len", torch.add, vid_len, txt_len)
        concat = cache("mm_pnp", na.concat_idx, vid_len, txt_len)[0]

        vid_frames = cache("vid_frames", lambda vid_shape: vid_shape[:, 0], vid_shape)
        vid_len_2d = cache(
            "vid_len_2d",
            lambda vid_shape: vid_shape[:, 1:].prod(-1).repeat_interleave(vid_shape[:, 0]),
            vid_shape,
        )
        txt_len_2d = cache("txt_len_2d", torch.repeat_interleave, txt_len, vid_frames)
        all_len_2d = cache("all_len_2d", torch.add, vid_len_2d, txt_len_2d)

        # Use of txt_len instead of txt_len_2d is intentional.
        concat2d, _ = cache("mm_pnp_2d", na.repeat_concat_idx, vid_len_2d, txt_len, vid_frames)

        vid_out = self.attn(
            q=vid_q.bfloat16(),
            k=concat2d(vid_k, txt_k).bfloat16(),
            v=concat2d(vid_v, txt_v).bfloat16(),
            cu_seqlens_q=cache("vid_seqlens_2d", na.cu_seqlens, vid_len_2d),
            cu_seqlens_k=cache("mm_seqlens_2d", na.cu_seqlens, all_len_2d),
            max_seqlen_q=cache("vid_maxlen_2d", na.max_seqlen, vid_len_2d),
            max_seqlen_k=cache("mm_maxlen_2d", na.max_seqlen, all_len_2d),
        ).type_as(vid_q)

        txt_out = self.attn(
            q=txt_q.bfloat16(),
            k=concat(vid_k, txt_k).bfloat16(),
            v=concat(vid_v, txt_v).bfloat16(),
            cu_seqlens_q=cache("txt_seqlens", na.cu_seqlens, txt_len),
            cu_seqlens_k=cache("mm_seqlens", na.cu_seqlens, all_len),
            max_seqlen_q=cache("txt_maxlen", na.max_seqlen, txt_len),
            max_seqlen_k=cache("mm_maxlen", na.max_seqlen, all_len),
        ).type_as(txt_q)

        vid_out = rearrange(vid_out, "l h d -> l (h d)")
        txt_out = rearrange(txt_out, "l h d -> l (h d)")
        vid_out = gather_heads_scatter_seq(vid_out, head_dim=1, seq_dim=0)
        txt_out = gather_heads_scatter_seq(txt_out, head_dim=1, seq_dim=0)

        vid_out, txt_out = self.proj_out(vid_out, txt_out)
        return vid_out, txt_out


class NaMMWindowAttention(NaMMAttention):
    def __init__(
        self,
        *args,
        window: Union[int, Tuple[int, int, int]],
        window_method: str,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.window = _triple(window)
        self.window_method = window_method
        assert all(map(lambda v: isinstance(v, int) and v >= 0, self.window))

        self.window_op = get_window_op(window_method)

        def make_window(x: torch.Tensor):
            t, h, w, _ = x.shape
            window_slices = self.window_op((t, h, w), self.window)
            return [x[st, sh, sw] for (st, sh, sw) in window_slices]

        self.make_window_fn = make_window

    def forward(
        self,
        vid: torch.FloatTensor,  # l c
        txt: torch.FloatTensor,  # l c
        vid_shape: torch.LongTensor,  # b 3
        txt_shape: torch.LongTensor,  # b 1
        cache: Cache,
    ) -> Tuple[
        torch.FloatTensor,
        torch.FloatTensor,
    ]:
        vid_qkv, txt_qkv = self.proj_qkv(vid, txt)
        vid_qkv = gather_seq_scatter_heads_qkv(
            vid_qkv,
            seq_dim=0,
            qkv_shape=vid_shape,
            cache=cache.namespace("vid"),
        )
        txt_qkv = gather_seq_scatter_heads_qkv(
            txt_qkv,
            seq_dim=0,
            qkv_shape=txt_shape,
            cache=cache.namespace("txt"),
        )
        vid_qkv = rearrange(vid_qkv, "l (o h d) -> l o h d", o=3, d=self.head_dim)
        txt_qkv = rearrange(txt_qkv, "l (o h d) -> l o h d", o=3, d=self.head_dim)

        vid_q, vid_k, vid_v = vid_qkv.unbind(1)
        txt_q, txt_k, txt_v = txt_qkv.unbind(1)

        vid_q, txt_q = self.norm_q(vid_q, txt_q)
        vid_k, txt_k = self.norm_k(vid_k, txt_k)

        # global rope
        if self.rope:
            if self.rope.mm:
                vid_q, vid_k, txt_q, txt_k = self.rope(
                    vid_q, vid_k, vid_shape, txt_q, txt_k, txt_shape, cache
                )
            else:
                vid_q, vid_k = self.rope(vid_q, vid_k, vid_shape, cache)

        # re-org the input seq for window attn
        cache_win = cache.namespace(f"{self.window_method}_{self.window}")

        window_partition, window_reverse, window_shape, window_count = cache_win(
            "win_transform", partial(na.window_idx, window_fn=self.make_window_fn), vid_shape
        )

        vid_len = cache("vid_len", na.get_seqlen, vid_shape)
        txt_len = cache("txt_len", na.get_seqlen, txt_shape)
        all_len = cache("all_len", torch.add, vid_len, txt_len)
        concat = cache("mm_pnp", na.concat_idx, vid_len, txt_len)[0]

        vid_len_win = cache_win("vid_len", na.get_seqlen, window_shape)
        txt_len_win = cache_win("txt_len", torch.repeat_interleave, txt_len, window_count)
        all_len_win = cache_win("all_len", torch.add, vid_len_win, txt_len_win)

        concat_win, _ = cache_win(
            "mm_pnp", na.repeat_concat_idx, vid_len_win, txt_len, window_count
        )

        vid_out = self.attn(
            q=window_partition(vid_q).bfloat16(),
            k=concat_win(window_partition(vid_k), txt_k).bfloat16(),
            v=concat_win(window_partition(vid_v), txt_v).bfloat16(),
            cu_seqlens_q=cache_win("vid_seqlens", na.cu_seqlens, vid_len_win),
            cu_seqlens_k=cache_win("mm_seqlens", na.cu_seqlens, all_len_win),
            max_seqlen_q=cache_win("vid_maxlen", na.max_seqlen, vid_len_win),
            max_seqlen_k=cache_win("mm_maxlen", na.max_seqlen, all_len_win),
        ).type_as(vid_q)

        # As we do full attn for txt, just use the reorganized input seq
        txt_out = self.attn(
            q=txt_q.bfloat16(),
            k=concat(vid_k, txt_k).bfloat16(),
            v=concat(vid_v, txt_v).bfloat16(),
            cu_seqlens_q=cache("txt_seqlens", na.cu_seqlens, txt_len),
            cu_seqlens_k=cache("mm_seqlens", na.cu_seqlens, all_len),
            max_seqlen_q=cache("txt_maxlen", na.max_seqlen, txt_len),
            max_seqlen_k=cache("mm_maxlen", na.max_seqlen, all_len),
        ).type_as(txt_q)

        vid_out = rearrange(vid_out, "l h d -> l (h d)")
        txt_out = rearrange(txt_out, "l h d -> l (h d)")

        vid_out = window_reverse(vid_out)
        vid_out = gather_heads_scatter_seq(vid_out, head_dim=1, seq_dim=0)
        txt_out = gather_heads_scatter_seq(txt_out, head_dim=1, seq_dim=0)

        vid_out, txt_out = self.proj_out(vid_out, txt_out)
        return vid_out, txt_out
