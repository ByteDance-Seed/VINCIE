import torch
from einops import rearrange
from torch import nn

from common.cache_v2 import Cache

from ... import na
from ...attention import FlashAttentionVarlen
from ...normalization import norm_layer_type


class NaCrossAttention(nn.Module):
    def __init__(
        self,
        vid_dim: int,
        txt_dim: int,
        heads: int,
        head_dim: int,
        qk_bias: bool,
        qk_norm: norm_layer_type,
        qk_norm_eps: float,
        **kwargs,
    ):
        super().__init__()
        inner_dim = heads * head_dim
        self.heads = heads
        self.head_dim = head_dim
        self.proj_q = nn.Linear(vid_dim, inner_dim, bias=qk_bias)
        self.proj_kv = nn.Linear(txt_dim, inner_dim * 2, bias=qk_bias)
        self.proj_out = nn.Linear(inner_dim, vid_dim)
        self.norm_q = qk_norm(dim=head_dim, eps=qk_norm_eps, elementwise_affine=True)
        self.norm_k = qk_norm(dim=head_dim, eps=qk_norm_eps, elementwise_affine=True)

        self.attn = FlashAttentionVarlen()

    def forward(
        self,
        vid: torch.FloatTensor,  # l c
        txt: torch.FloatTensor,  # l c
        vid_shape: torch.LongTensor,  # b 3
        txt_shape: torch.LongTensor,  # b 1
        cache: Cache,
    ) -> torch.FloatTensor:

        vid_q, txt_kv = self.proj_q(vid), self.proj_kv(txt)
        vid_q = rearrange(vid_q, "l (h d) -> l h d", d=self.head_dim)
        txt_kv = rearrange(txt_kv, "l (o h d) -> l o h d", o=2, d=self.head_dim)

        txt_k, txt_v = txt_kv.unbind(1)

        vid_q = self.norm_q(vid_q)
        txt_k = self.norm_k(txt_k)

        vid_len = cache("vid_len", na.get_seqlen, vid_shape)
        txt_len = cache("txt_len", na.get_seqlen, txt_shape)

        vid_out = self.attn(
            q=vid_q.bfloat16(),
            k=txt_k.bfloat16(),
            v=txt_v.bfloat16(),
            cu_seqlens_q=cache("vid_seqlens", na.cu_seqlens, vid_len),
            cu_seqlens_k=cache("txt_seqlens", na.cu_seqlens, txt_len),
            max_seqlen_q=cache("vid_maxlen", na.max_seqlen, vid_len),
            max_seqlen_k=cache("txt_maxlen", na.max_seqlen, txt_len),
        ).type_as(vid_q)

        vid_out = rearrange(vid_out, "l h d -> l (h d)")
        vid_out = self.proj_out(vid_out)
        return vid_out
