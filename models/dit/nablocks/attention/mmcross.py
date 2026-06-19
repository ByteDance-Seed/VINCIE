import torch
from einops import rearrange
from torch import nn

from common.cache_v2 import Cache
from common.distributed.ops import gather_heads_scatter_seq, gather_seq_scatter_heads_qkv

from ... import na
from ...attention import FlashAttentionVarlen
from ...mm import MMArg, MMModule
from ...normalization import norm_layer_type


class NaMMCrossAttention(nn.Module):
    def __init__(
        self,
        vid_dim: int,
        txt_dim: int,
        heads: int,
        head_dim: int,
        qk_bias: bool,
        qk_norm: norm_layer_type,
        qk_norm_eps: float,
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

        self.attn = FlashAttentionVarlen()

    def forward(
        self,
        vid: torch.FloatTensor,  # l c
        txt: torch.FloatTensor,  # l c
        vid_shape: torch.LongTensor,  # b 3
        txt_shape: torch.LongTensor,  # b 1
        cache: Cache,
    ) -> torch.FloatTensor:

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

        vid_len = cache("vid_len", na.get_seqlen, vid_shape)
        txt_len = cache("txt_len", na.get_seqlen, txt_shape)
        all_len = cache("all_len", torch.add, vid_len, txt_len)
        concat = cache("mm_pnp", na.concat_idx, vid_len, txt_len)[0]

        vid_out = self.attn(
            q=vid_q.bfloat16(),
            k=txt_k.bfloat16(),
            v=txt_v.bfloat16(),
            cu_seqlens_q=cache("vid_seqlens", na.cu_seqlens, vid_len),
            cu_seqlens_k=cache("txt_seqlens", na.cu_seqlens, txt_len),
            max_seqlen_q=cache("vid_maxlen", na.max_seqlen, vid_len),
            max_seqlen_k=cache("txt_maxlen", na.max_seqlen, txt_len),
        ).type_as(vid_q)

        txt_out = self.attn(
            q=txt_q.bfloat16(),
            k=concat(vid_k, txt_k).bfloat16(),
            v=concat(vid_v, txt_v).bfloat16(),
            cu_seqlens_q=cache("txt_seqlens", na.cu_seqlens, txt_len),
            cu_seqlens_k=cache("mm_seqlens", na.cu_seqlens, all_len),
            max_seqlen_q=cache("txt_maxlen", na.max_seqlen, txt_len),
            max_seqlen_k=cache("mm_maxlen", na.max_seqlen, all_len),
        ).type_as(vid_q)

        vid_out = rearrange(vid_out, "l h d -> l (h d)")
        txt_out = rearrange(txt_out, "l h d -> l (h d)")
        vid_out = gather_heads_scatter_seq(vid_out, head_dim=1, seq_dim=0)
        txt_out = gather_heads_scatter_seq(txt_out, head_dim=1, seq_dim=0)

        vid_out, txt_out = self.proj_out(vid_out, txt_out)
        return vid_out, txt_out
