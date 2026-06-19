from functools import partial
from typing import Optional, Tuple
import torch
import turbox
from einops import rearrange
from torch.nn import functional as F

from common.cache_v2 import Cache
from common.distributed.ops import gather_heads_scatter_seq, gather_seq_scatter_heads_qkv

from ... import na
from .mmattn import NaMMAttention, NaMMSpaceAttention, NaMMWindowAttention


class FusedNaMMAttention(NaMMAttention):
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

        if self.rope.mm:
            vid_freqs, txt_freqs = cache(
                f"{self.rope.__class__.__name__}_freqs",
                self.rope.get_freqs,
                vid_shape,
                txt_shape,
            )
        else:
            vid_freqs = cache(
                f"{self.rope.__class__.__name__}_freqs",
                self.rope.get_freqs,
                vid_shape,
            )

        vid_len = cache("vid_len", na.get_seqlen, vid_shape)
        txt_len = cache("txt_len", na.get_seqlen, txt_shape)
        all_len = cache("all_len", torch.add, vid_len, txt_len)

        vid_q, txt_q, mm_vid_txt_k, mm_vid_txt_v = turbox.mmdit_v2_full_block_func(
            vid=vid_qkv,
            txt=txt_qkv,
            vid_q_gamma=getattr(self.norm_q.get_module("vid"), "weight", None),
            vid_k_gamma=getattr(self.norm_k.get_module("vid"), "weight", None),
            txt_q_gamma=getattr(self.norm_q.get_module("txt"), "weight", None),
            txt_k_gamma=getattr(self.norm_k.get_module("txt"), "weight", None),
            vid_q_bias=getattr(self.norm_q.get_module("vid"), "bias", None),
            vid_k_bias=getattr(self.norm_k.get_module("vid"), "bias", None),
            txt_q_bias=getattr(self.norm_q.get_module("txt"), "bias", None),
            txt_k_bias=getattr(self.norm_k.get_module("txt"), "bias", None),
            vid_sines=cache("vid_sines", torch.sin, vid_freqs),
            vid_coses=cache("vid_coses", torch.cos, vid_freqs),
            txt_sines=cache("txt_sines", lambda: txt_freqs.sin() if self.rope.mm else None),
            txt_coses=cache("txt_coses", lambda: txt_freqs.cos() if self.rope.mm else None),
            vid_lens=vid_len.int(),
            txt_lens=txt_len.int(),
            txt_rope_index=getattr(self.rope, "start_index", 0),
            rms=not hasattr(self.norm_q.get_module("vid"), "bias"),
            eps=self.norm_q.get_module("vid").eps,
        )

        vid_out = self.attn(
            q=vid_q.bfloat16(),
            k=mm_vid_txt_k.bfloat16(),
            v=mm_vid_txt_v.bfloat16(),
            cu_seqlens_q=cache("vid_seqlens", na.cu_seqlens, vid_len),
            cu_seqlens_k=cache("mm_seqlens", na.cu_seqlens, all_len),
            max_seqlen_q=cache("vid_maxlen", na.max_seqlen, vid_len),
            max_seqlen_k=cache("mm_maxlen", na.max_seqlen, all_len),
        ).type_as(vid_qkv)

        txt_out = self.attn(
            q=txt_q.bfloat16(),
            k=mm_vid_txt_k.bfloat16(),
            v=mm_vid_txt_v.bfloat16(),
            cu_seqlens_q=cache("txt_seqlens", na.cu_seqlens, txt_len),
            cu_seqlens_k=cache("mm_seqlens", na.cu_seqlens, all_len),
            max_seqlen_q=cache("txt_maxlen", na.max_seqlen, txt_len),
            max_seqlen_k=cache("mm_maxlen", na.max_seqlen, all_len),
        ).type_as(vid_qkv)

        vid_out = rearrange(vid_out, "l h d -> l (h d)")
        txt_out = rearrange(txt_out, "l h d -> l (h d)")

        vid_out = gather_heads_scatter_seq(vid_out, head_dim=1, seq_dim=0)
        txt_out = gather_heads_scatter_seq(txt_out, head_dim=1, seq_dim=0)

        vid_out, txt_out = self.proj_out(vid_out, txt_out)
        return vid_out, txt_out


class FusedNaMMSpaceAttention(NaMMSpaceAttention):
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

        if self.rope.mm:
            vid_freqs, txt_freqs = cache(
                f"{self.rope.__class__.__name__}_freqs",
                self.rope.get_freqs,
                vid_shape,
                txt_shape,
            )
        else:
            vid_freqs = cache(
                f"{self.rope.__class__.__name__}_freqs",
                self.rope.get_freqs,
                vid_shape,
            )

        vid_len = cache("vid_len", na.get_seqlen, vid_shape)
        txt_len = cache("txt_len", na.get_seqlen, txt_shape)
        all_len = cache("all_len", torch.add, vid_len, txt_len)
        vid_shape_i = cache("vid_shape_i", lambda vid_shape: vid_shape.int(), vid_shape)

        vid_frames = cache("vid_frames", lambda vid_shape: vid_shape[:, 0], vid_shape)
        vid_len_2d = cache(
            "vid_len_2d",
            lambda vid_shape: vid_shape[:, 1:].prod(-1).repeat_interleave(vid_shape[:, 0]),
            vid_shape,
        )
        txt_len_2d = cache("txt_len_2d", torch.repeat_interleave, txt_len, vid_frames)
        all_len_2d = cache("all_len_2d", torch.add, vid_len_2d, txt_len_2d)

        txt_offsets = cache(
            "txt_offsets",
            lambda txt_len, vid_shape: (F.pad(txt_len, (1, 0))[:-1] + vid_shape.prod(-1))
            .cumsum(0)
            .repeat_interleave(vid_shape[:, 0])
            .int(),
            txt_len,
            vid_shape,
        )
        txt_total_len_2d = cache(
            "txt_total_len_2d", lambda txt_len_2d: txt_len_2d.sum().int().item(), txt_len_2d
        )

        vid_q, txt_q, vid_txt_k, vid_txt_v, vid_txt_k_2d, vid_txt_v_2d = (
            turbox.mmdit_v2_temp_win_block_func(
                vid=vid_qkv,
                txt=txt_qkv,
                vid_q_gamma=getattr(self.norm_q.get_module("vid"), "weight", None),
                vid_k_gamma=getattr(self.norm_k.get_module("vid"), "weight", None),
                txt_q_gamma=getattr(self.norm_q.get_module("txt"), "weight", None),
                txt_k_gamma=getattr(self.norm_k.get_module("txt"), "weight", None),
                vid_q_bias=getattr(self.norm_q.get_module("vid"), "bias", None),
                vid_k_bias=getattr(self.norm_k.get_module("vid"), "bias", None),
                txt_q_bias=getattr(self.norm_q.get_module("txt"), "bias", None),
                txt_k_bias=getattr(self.norm_k.get_module("txt"), "bias", None),
                vid_sines=cache("vid_sines", torch.sin, vid_freqs),
                vid_coses=cache("vid_coses", torch.cos, vid_freqs),
                txt_sines=cache("txt_sines", lambda: txt_freqs.sin() if self.rope.mm else None),
                txt_coses=cache("txt_coses", lambda: txt_freqs.cos() if self.rope.mm else None),
                vid_lens=vid_len.int(),
                txt_lens=txt_len.int(),
                vid_shape=vid_shape_i,
                temp_win_vid_lens=vid_len_2d.int(),
                temp_win_txt_lens=txt_len_2d.int(),
                temp_win_txt_offsets=txt_offsets,
                temp_win_sizes=None,
                temp_win_txt_total_len=txt_total_len_2d,
                temp_win_size=1,  # space block is a special case of temp_win with temp_win_size=1.
                txt_rope_index=getattr(self.rope, "start_index", 0),
                rms=not hasattr(self.norm_q.get_module("vid"), "bias"),
                eps=self.norm_q.get_module("vid").eps,
            )
        )

        vid_out = self.attn(
            q=vid_q.bfloat16(),
            k=vid_txt_k_2d.bfloat16(),
            v=vid_txt_v_2d.bfloat16(),
            cu_seqlens_q=cache("vid_seqlens_2d", na.cu_seqlens, vid_len_2d),
            cu_seqlens_k=cache("mm_seqlens_2d", na.cu_seqlens, all_len_2d),
            max_seqlen_q=cache("vid_maxlen_2d", na.max_seqlen, vid_len_2d),
            max_seqlen_k=cache("mm_maxlen_2d", na.max_seqlen, all_len_2d),
        ).type_as(vid_qkv)

        txt_out = self.attn(
            q=txt_q.bfloat16(),
            k=vid_txt_k.bfloat16(),
            v=vid_txt_v.bfloat16(),
            cu_seqlens_q=cache("txt_seqlens", na.cu_seqlens, txt_len),
            cu_seqlens_k=cache("mm_seqlens", na.cu_seqlens, all_len),
            max_seqlen_q=cache("txt_maxlen", na.max_seqlen, txt_len),
            max_seqlen_k=cache("mm_maxlen", na.max_seqlen, all_len),
        ).type_as(txt_qkv)

        vid_out = rearrange(vid_out, "l h d -> l (h d)")
        txt_out = rearrange(txt_out, "l h d -> l (h d)")

        vid_out = gather_heads_scatter_seq(vid_out, head_dim=1, seq_dim=0)
        txt_out = gather_heads_scatter_seq(txt_out, head_dim=1, seq_dim=0)

        vid_out, txt_out = self.proj_out(vid_out, txt_out)
        return vid_out, txt_out


class FusedNaMMWindowAttention(NaMMWindowAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.window_method in ["win", "swin", "win_by_size"]
        if self.window_method == "win_by_size":
            assert self.window[1] == self.window[2] == 0, "Only support temp_win by_size"

        def make_window(x: torch.Tensor):
            t, h, w, _ = x.shape
            window_slices = self.window_op((t, h, w), self.window, skip_empty=False)
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

        # global rope
        if self.rope.mm:
            vid_freqs, txt_freqs = cache(
                f"{self.rope.__class__.__name__}_freqs",
                self.rope.get_freqs,
                vid_shape,
                txt_shape,
            )
        else:
            vid_freqs = cache(
                f"{self.rope.__class__.__name__}_freqs",
                self.rope.get_freqs,
                vid_shape,
            )
            txt_freqs = None

        if self.window_method in ["win", "swin"]:
            forward_fn = self.by_number_forward
        elif self.window_method in ["win_by_size"]:
            forward_fn = self.by_size_forward
        else:
            raise NotImplementedError

        return forward_fn(
            vid_qkv=vid_qkv,
            txt_qkv=txt_qkv,
            vid_freqs=vid_freqs,
            txt_freqs=txt_freqs,
            vid_shape=vid_shape,
            txt_shape=txt_shape,
            cache=cache,
        )

    def by_size_forward(
        self,
        vid_qkv: torch.FloatTensor,  # l c
        txt_qkv: torch.FloatTensor,  # l c
        vid_freqs: torch.FloatTensor,  # l c
        txt_freqs: Optional[torch.FloatTensor],  # l c
        vid_shape: torch.LongTensor,  # b 3
        txt_shape: torch.LongTensor,  # b 1
        cache: Cache,
    ) -> Tuple[
        torch.FloatTensor,
        torch.FloatTensor,
    ]:
        # re-org the input seq for window attn
        cache_win = cache.namespace(f"{self.window_method}_{self.window}")

        _, _, window_shape, window_count = cache_win(
            "win_transform", partial(na.window_idx, window_fn=self.make_window_fn), vid_shape
        )

        vid_len = cache("vid_len", na.get_seqlen, vid_shape)
        txt_len = cache("txt_len", na.get_seqlen, txt_shape)
        all_len = cache("all_len", torch.add, vid_len, txt_len)
        vid_shape_i = cache("vid_shape_i", lambda vid_shape: vid_shape.int(), vid_shape)

        win_vid_len = cache_win("vid_len", na.get_seqlen, window_shape)
        win_txt_len = cache_win("txt_len", torch.repeat_interleave, txt_len, window_count)
        win_all_len = cache_win("all_len", torch.add, win_vid_len, win_txt_len)
        win_txt_offsets = cache_win(
            "txt_offsets",
            lambda txt_len, vid_shape, window_count: (
                F.pad(txt_len, (1, 0))[:-1] + vid_shape.prod(-1)
            )
            .cumsum(0)
            .repeat_interleave(window_count)
            .int(),
            txt_len,
            vid_shape,
            window_count,
        )
        win_sizes = cache_win(
            "win_sizes",
            lambda vid_shape, window_count: torch.ceil(vid_shape[:, 0] / window_count).int(),
            vid_shape,
            window_count,
        )
        win_txt_total_len = cache_win(
            "txt_total_len", lambda win_txt_len: win_txt_len.sum().int().item(), win_txt_len
        )

        vid_q, txt_q, vid_txt_k, vid_txt_v, win_vid_txt_k, win_vid_txt_v = (
            turbox.mmdit_v2_temp_win_block_func(
                vid=vid_qkv,
                txt=txt_qkv,
                vid_q_gamma=getattr(self.norm_q.get_module("vid"), "weight", None),
                vid_k_gamma=getattr(self.norm_k.get_module("vid"), "weight", None),
                txt_q_gamma=getattr(self.norm_q.get_module("txt"), "weight", None),
                txt_k_gamma=getattr(self.norm_k.get_module("txt"), "weight", None),
                vid_q_bias=getattr(self.norm_q.get_module("vid"), "bias", None),
                vid_k_bias=getattr(self.norm_k.get_module("vid"), "bias", None),
                txt_q_bias=getattr(self.norm_q.get_module("txt"), "bias", None),
                txt_k_bias=getattr(self.norm_k.get_module("txt"), "bias", None),
                vid_sines=cache("vid_sines", torch.sin, vid_freqs),
                vid_coses=cache("vid_coses", torch.cos, vid_freqs),
                txt_sines=cache("txt_sines", lambda: txt_freqs.sin() if self.rope.mm else None),
                txt_coses=cache("txt_coses", lambda: txt_freqs.cos() if self.rope.mm else None),
                vid_lens=vid_len.int(),
                txt_lens=txt_len.int(),
                vid_shape=vid_shape_i,
                temp_win_vid_lens=win_vid_len.int(),
                temp_win_txt_lens=win_txt_len.int(),
                temp_win_txt_offsets=win_txt_offsets,
                temp_win_sizes=win_sizes,
                temp_win_txt_total_len=win_txt_total_len,
                temp_win_size=self.window[0],
                txt_rope_index=getattr(self.rope, "start_index", 0),
                rms=not hasattr(self.norm_q.get_module("vid"), "bias"),
                eps=self.norm_q.get_module("vid").eps,
            )
        )

        vid_out = self.attn(
            q=vid_q.bfloat16(),
            k=win_vid_txt_k.bfloat16(),
            v=win_vid_txt_v.bfloat16(),
            cu_seqlens_q=cache_win("vid_seqlens", na.cu_seqlens, win_vid_len),
            cu_seqlens_k=cache_win("mm_seqlens", na.cu_seqlens, win_all_len),
            max_seqlen_q=cache_win("vid_maxlen", na.max_seqlen, win_vid_len),
            max_seqlen_k=cache_win("mm_maxlen", na.max_seqlen, win_all_len),
        ).type_as(vid_qkv)

        txt_out = self.attn(
            q=txt_q.bfloat16(),
            k=vid_txt_k.bfloat16(),
            v=vid_txt_v.bfloat16(),
            cu_seqlens_q=cache("txt_seqlens", na.cu_seqlens, txt_len),
            cu_seqlens_k=cache("mm_seqlens", na.cu_seqlens, all_len),
            max_seqlen_q=cache("txt_maxlen", na.max_seqlen, txt_len),
            max_seqlen_k=cache("mm_maxlen", na.max_seqlen, all_len),
        ).type_as(txt_qkv)

        vid_out = rearrange(vid_out, "l h d -> l (h d)")
        txt_out = rearrange(txt_out, "l h d -> l (h d)")

        vid_out = gather_heads_scatter_seq(vid_out, head_dim=1, seq_dim=0)
        txt_out = gather_heads_scatter_seq(txt_out, head_dim=1, seq_dim=0)

        vid_out, txt_out = self.proj_out(vid_out, txt_out)
        return vid_out, txt_out

    def by_number_forward(
        self,
        vid_qkv: torch.FloatTensor,  # l c
        txt_qkv: torch.FloatTensor,  # l c
        vid_freqs: torch.FloatTensor,  # l c
        txt_freqs: Optional[torch.FloatTensor],  # l c
        vid_shape: torch.LongTensor,  # b 3
        txt_shape: torch.LongTensor,  # b 1
        cache: Cache,
    ) -> Tuple[
        torch.FloatTensor,
        torch.FloatTensor,
    ]:
        # re-org the input seq for window attn
        cache_win = cache.namespace(f"{self.window_method}_{self.window}")

        _, _, window_shape, window_count = cache_win(
            "win_transform", partial(na.window_idx, window_fn=self.make_window_fn), vid_shape
        )

        vid_len = cache("vid_len", na.get_seqlen, vid_shape)
        txt_len = cache("txt_len", na.get_seqlen, txt_shape)
        all_len = cache("all_len", torch.add, vid_len, txt_len)
        vid_shape_i = cache("vid_shape_i", lambda vid_shape: vid_shape.int(), vid_shape)

        win_vid_len = cache_win("vid_len", na.get_seqlen, window_shape)
        win_txt_len = cache_win(
            "txt_len",
            lambda txt_len, window_count, win_vid_len: (
                txt_len.repeat_interleave(window_count) * (win_vid_len > 0).type_as(txt_len)
            ),
            txt_len,
            window_count,
            win_vid_len,
        )
        win_all_len = cache_win("all_len", torch.add, win_vid_len, win_txt_len)
        win_txt_total_len = cache_win(
            "txt_total_len", lambda win_txt_len: win_txt_len.sum().item(), win_txt_len
        )
        win_vid_num = cache_win(
            "vid_num",
            lambda win_vid_len, vid_shape: (win_vid_len > 0)
            .reshape(vid_shape.shape[0], -1)
            .sum(1)
            .int(),
            win_vid_len,
            vid_shape,
        )

        win_vid_offsets = cache_win(
            "vid_offsets",
            partial(na.cu_seqlens, skip_empty=False),
            win_vid_len,
        )
        win_txt_offsets = cache_win(
            "txt_offsets",
            lambda txt_len, vid_shape, window_count: (
                F.pad(txt_len, (1, 0))[:-1] + vid_shape.prod(-1)
            )
            .cumsum(0)
            .repeat_interleave(window_count)
            .int(),
            txt_len,
            vid_shape,
            window_count,
        )
        win_vid_txt_offsets = cache_win(
            "vid_txt_offsets", partial(na.cu_seqlens, skip_empty=False), win_all_len
        )

        win_vid_q, txt_q, vid_txt_k, vid_txt_v, win_vid_txt_k, win_vid_txt_v = (
            turbox.mmdit_v2_win_block_func(
                vid=vid_qkv,
                txt=txt_qkv,
                vid_q_gamma=getattr(self.norm_q.get_module("vid"), "weight", None),
                vid_k_gamma=getattr(self.norm_k.get_module("vid"), "weight", None),
                txt_q_gamma=getattr(self.norm_q.get_module("txt"), "weight", None),
                txt_k_gamma=getattr(self.norm_k.get_module("txt"), "weight", None),
                vid_q_bias=getattr(self.norm_q.get_module("vid"), "bias", None),
                vid_k_bias=getattr(self.norm_k.get_module("vid"), "bias", None),
                txt_q_bias=getattr(self.norm_q.get_module("txt"), "bias", None),
                txt_k_bias=getattr(self.norm_k.get_module("txt"), "bias", None),
                vid_sines=cache("vid_sines", torch.sin, vid_freqs),
                vid_coses=cache("vid_coses", torch.cos, vid_freqs),
                txt_sines=cache("txt_sines", lambda: txt_freqs.sin() if self.rope.mm else None),
                txt_coses=cache("txt_coses", lambda: txt_freqs.cos() if self.rope.mm else None),
                vid_lens=vid_len.int(),
                txt_lens=txt_len.int(),
                win_vid_lens=win_vid_len.int(),
                win_txt_lens=win_txt_len.int(),
                vid_shape=vid_shape_i,
                win_vid_nums=win_vid_num,
                win_vid_offsets=win_vid_offsets,
                win_txt_offsets=win_txt_offsets,
                win_vid_txt_offsets=win_vid_txt_offsets,
                window=self.window,
                win_txt_total_len=win_txt_total_len,
                txt_rope_index=getattr(self.rope, "start_index", 0),
                shift=self.window_method == "swin",
                rms=not hasattr(self.norm_q.get_module("vid"), "bias"),
                eps=self.norm_q.get_module("vid").eps,
            )
        )

        win_vid_out = self.attn(
            q=win_vid_q.bfloat16(),
            k=win_vid_txt_k.bfloat16(),
            v=win_vid_txt_v.bfloat16(),
            cu_seqlens_q=cache_win("vid_seqlens", na.cu_seqlens, win_vid_len),
            cu_seqlens_k=cache_win("mm_seqlens", na.cu_seqlens, win_all_len),
            max_seqlen_q=cache_win("vid_maxlen", na.max_seqlen, win_vid_len),
            max_seqlen_k=cache_win("mm_maxlen", na.max_seqlen, win_all_len),
        ).type_as(vid_qkv)

        # As we do full attn for txt, just use the reorganized input seq
        txt_out = self.attn(
            q=txt_q.bfloat16(),
            k=vid_txt_k.bfloat16(),
            v=vid_txt_v.bfloat16(),
            cu_seqlens_q=cache("txt_seqlens", na.cu_seqlens, txt_len),
            cu_seqlens_k=cache("mm_seqlens", na.cu_seqlens, all_len),
            max_seqlen_q=cache("txt_maxlen", na.max_seqlen, txt_len),
            max_seqlen_k=cache("mm_maxlen", na.max_seqlen, all_len),
        ).type_as(txt_qkv)

        vid_out = turbox.mmdit_v2_win_block_reverse_func(
            win_vid=win_vid_out,
            vid_lens=vid_len.int(),
            vid_shape=vid_shape_i,
            win_vid_offsets=win_vid_offsets.int(),
            window=self.window,
            shift=self.window_method == "swin",
        )

        vid_out = rearrange(vid_out, "l h d -> l (h d)")
        txt_out = rearrange(txt_out, "l h d -> l (h d)")

        vid_out = gather_heads_scatter_seq(vid_out, head_dim=1, seq_dim=0)
        txt_out = gather_heads_scatter_seq(txt_out, head_dim=1, seq_dim=0)

        vid_out, txt_out = self.proj_out(vid_out, txt_out)
        return vid_out, txt_out
