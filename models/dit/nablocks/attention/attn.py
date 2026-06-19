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
from ...normalization import norm_layer_type
from ...rope import get_na_rope
from ...window import get_window_op


class NaSelfAttention(nn.Module):
    def __init__(
        self,
        vid_dim: int,
        heads: int,
        head_dim: int,
        qk_bias: bool,
        qk_norm: norm_layer_type,
        qk_norm_eps: float,
        rope_type: Optional[str],
        rope_dim: int,
        **kwargs,
    ):
        super().__init__()
        inner_dim = heads * head_dim
        self.heads = heads
        self.head_dim = head_dim
        self.proj_qkv = nn.Linear(vid_dim, 3 * inner_dim, bias=qk_bias)
        self.proj_out = nn.Linear(inner_dim, vid_dim)
        self.norm_q = qk_norm(dim=head_dim, eps=qk_norm_eps, elementwise_affine=True)
        self.norm_k = qk_norm(dim=head_dim, eps=qk_norm_eps, elementwise_affine=True)
        self.rope = get_na_rope(rope_type=rope_type, dim=rope_dim)
        self.attn = FlashAttentionVarlen()

    def forward(
        self,
        vid: torch.FloatTensor,  # l c
        vid_shape: torch.LongTensor,  # b 3
        cache: Cache,
    ) -> torch.FloatTensor:

        vid_qkv = self.proj_qkv(vid)
        vid_qkv = gather_seq_scatter_heads_qkv(
            vid_qkv, seq_dim=0, qkv_shape=vid_shape, cache=cache.namespace("vid")
        )
        vid_qkv = rearrange(vid_qkv, "l (o h d) -> l o h d", o=3, d=self.head_dim)

        vid_q, vid_k, vid_v = vid_qkv.unbind(1)

        vid_q = self.norm_q(vid_q)
        vid_k = self.norm_k(vid_k)

        if self.rope:
            vid_q, vid_k = self.rope(vid_q, vid_k, vid_shape, cache)

        vid_len = cache("vid_len", na.get_seqlen, vid_shape)

        vid_out = self.attn(
            q=vid_q.bfloat16(),
            k=vid_k.bfloat16(),
            v=vid_v.bfloat16(),
            cu_seqlens_q=cache("vid_seqlens", na.cu_seqlens, vid_len),
            cu_seqlens_k=cache("vid_seqlens", na.cu_seqlens, vid_len),
            max_seqlen_q=cache("vid_maxlen", na.max_seqlen, vid_len),
            max_seqlen_k=cache("vid_maxlen", na.max_seqlen, vid_len),
        ).type_as(vid_q)

        vid_out = rearrange(vid_out, "l h d -> l (h d)")
        vid_out = gather_heads_scatter_seq(vid_out, head_dim=1, seq_dim=0)
        vid_out = self.proj_out(vid_out)
        return vid_out


class NaSpaceAttention(NaSelfAttention):
    def forward(
        self,
        vid: torch.FloatTensor,  # l c
        vid_shape: torch.LongTensor,  # b 3
        cache: Cache,
    ) -> torch.FloatTensor:

        vid_qkv = self.proj_qkv(vid)
        vid_qkv = gather_seq_scatter_heads_qkv(
            vid_qkv, seq_dim=0, qkv_shape=vid_shape, cache=cache.namespace("vid")
        )
        vid_qkv = rearrange(vid_qkv, "l (o h d) -> l o h d", o=3, d=self.head_dim)

        vid_q, vid_k, vid_v = vid_qkv.unbind(1)

        vid_q = self.norm_q(vid_q)
        vid_k = self.norm_k(vid_k)

        if self.rope:
            vid_q, vid_k = self.rope(vid_q, vid_k, vid_shape, cache)

        vid_len_2d = cache(
            "vid_len_2d",
            lambda vid_shape: vid_shape[:, 1:].prod(-1).repeat_interleave(vid_shape[:, 0]),
            vid_shape,
        )

        vid_out = self.attn(
            q=vid_q.bfloat16(),
            k=vid_k.bfloat16(),
            v=vid_v.bfloat16(),
            cu_seqlens_q=cache("vid_seqlens_2d", na.cu_seqlens, vid_len_2d),
            cu_seqlens_k=cache("vid_seqlens_2d", na.cu_seqlens, vid_len_2d),
            max_seqlen_q=cache("vid_maxlen_2d", na.max_seqlen, vid_len_2d),
            max_seqlen_k=cache("vid_maxlen_2d", na.max_seqlen, vid_len_2d),
        ).type_as(vid_q)

        vid_out = rearrange(vid_out, "l h d -> l (h d)")
        vid_out = gather_heads_scatter_seq(vid_out, head_dim=1, seq_dim=0)
        vid_out = self.proj_out(vid_out)

        return vid_out


class NaTemporalAttention(NaSelfAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.temporal_op = get_window_op("win_by_size")
        self.window = (0, 1, 1)

        def make_temporal_window(x: torch.Tensor):
            t, h, w, _ = x.shape
            window_slices = self.temporal_op((t, h, w), self.window)
            return [x[st, sh, sw] for (st, sh, sw) in window_slices]

        self.make_temporal_window_fn = make_temporal_window

    def forward(
        self,
        vid: torch.FloatTensor,  # l c
        vid_shape: torch.LongTensor,  # b 3
        cache: Cache,
    ) -> torch.FloatTensor:

        vid_qkv = self.proj_qkv(vid)
        vid_qkv = gather_seq_scatter_heads_qkv(
            vid_qkv, seq_dim=0, qkv_shape=vid_shape, cache=cache.namespace("vid")
        )
        vid_qkv = rearrange(vid_qkv, "l (o h d) -> l o h d", o=3, d=self.head_dim)

        vid_q, vid_k, vid_v = vid_qkv.unbind(1)

        vid_q = self.norm_q(vid_q)
        vid_k = self.norm_k(vid_k)

        if self.rope:
            vid_q, vid_k = self.rope(vid_q, vid_k, vid_shape, cache)

        temporal_partition, temporal_reverse, temporal_shape, _ = cache(
            "time_transform",
            partial(na.window_idx, window_fn=self.make_temporal_window_fn),
            vid_shape,
        )

        vid_len_temporal = cache("vid_len_t", na.get_seqlen, temporal_shape)

        vid_out = self.attn(
            q=temporal_partition(vid_q).bfloat16(),
            k=temporal_partition(vid_k).bfloat16(),
            v=temporal_partition(vid_v).bfloat16(),
            cu_seqlens_q=cache("vid_seqlens_t", na.cu_seqlens, vid_len_temporal),
            cu_seqlens_k=cache("vid_seqlens_t", na.cu_seqlens, vid_len_temporal),
            max_seqlen_q=cache("vid_maxlen_t", na.max_seqlen, vid_len_temporal),
            max_seqlen_k=cache("vid_maxlen_t", na.max_seqlen, vid_len_temporal),
        ).type_as(vid_q)

        vid_out = rearrange(vid_out, "l h d -> l (h d)")
        vid_out = temporal_reverse(vid_out)
        vid_out = gather_heads_scatter_seq(vid_out, head_dim=1, seq_dim=0)
        vid_out = self.proj_out(vid_out)

        return vid_out


class NaWindowAttention(NaSelfAttention):
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
        self.window_op = get_window_op(window_method)
        assert all(map(lambda v: isinstance(v, int) and v >= 0, self.window))

        def make_window(x: torch.Tensor):
            t, h, w, _ = x.shape
            window_slices = self.window_op((t, h, w), self.window)
            return [x[st, sh, sw] for (st, sh, sw) in window_slices]

        self.make_window_fn = make_window

    def forward(
        self,
        vid: torch.FloatTensor,  # l c
        vid_shape: torch.LongTensor,  # b 3
        cache: Cache,
    ) -> Tuple[torch.FloatTensor,]:

        vid_qkv = self.proj_qkv(vid)
        vid_qkv = gather_seq_scatter_heads_qkv(
            vid_qkv, seq_dim=0, qkv_shape=vid_shape, cache=cache.namespace("vid")
        )
        vid_qkv = rearrange(vid_qkv, "l (o h d) -> l o h d", o=3, d=self.head_dim)

        vid_q, vid_k, vid_v = vid_qkv.unbind(1)

        vid_q = self.norm_q(vid_q)
        vid_k = self.norm_k(vid_k)

        if self.rope:
            vid_q, vid_k = self.rope(vid_q, vid_k, vid_shape, cache)

        cache_win = cache.namespace(f"{self.window_method}_{self.window}")

        window_partition, window_reverse, window_shape, _ = cache_win(
            "win_transform", partial(na.window_idx, window_fn=self.make_window_fn), vid_shape
        )
        vid_len_win = cache_win("vid_len", na.get_seqlen, window_shape)

        vid_out = self.attn(
            q=window_partition(vid_q).bfloat16(),
            k=window_partition(vid_k).bfloat16(),
            v=window_partition(vid_v).bfloat16(),
            cu_seqlens_q=cache_win("vid_seqlens", na.cu_seqlens, vid_len_win),
            cu_seqlens_k=cache_win("vid_seqlens", na.cu_seqlens, vid_len_win),
            max_seqlen_q=cache_win("vid_maxlen", na.max_seqlen, vid_len_win),
            max_seqlen_k=cache_win("vid_maxlen", na.max_seqlen, vid_len_win),
        ).type_as(vid_q)

        vid_out = rearrange(vid_out, "l h d -> l (h d)")
        vid_out = window_reverse(vid_out)
        vid_out = gather_heads_scatter_seq(vid_out, head_dim=1, seq_dim=0)

        vid_out = self.proj_out(vid_out)
        return vid_out
