import torch
from einops import rearrange
from torch import nn

from common.distributed.ops import (
    gather_heads_scatter_seq,
    gather_seq_scatter_heads,
    scatter_heads,
)

from ...attention import TorchAttention
from ...normalization import norm_layer_type


class CrossAttention(nn.Module):
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
        self.attn = TorchAttention()

    def forward(
        self,
        vid: torch.FloatTensor,  # b T H W c
        txt: torch.FloatTensor,  # b L c
        txt_mask: torch.BoolTensor,  # b L
    ) -> torch.FloatTensor:

        vid_q = self.proj_q(vid)
        txt_kv = self.proj_kv(txt)
        vid_q = gather_seq_scatter_heads(vid_q, seq_dim=2, head_dim=4)
        _, T, H, W, _ = vid_q.shape
        vid_q = rearrange(vid_q, "b T H W (h d) -> b h (T H W) d", d=self.head_dim)
        txt_kv = rearrange(txt_kv, "b L (o h d) -> o b h L d", o=2, d=self.head_dim)
        txt_kv = scatter_heads(txt_kv, dim=2)
        txt_k, txt_v = txt_kv.unbind()

        vid_q = self.norm_q(vid_q)
        txt_k = self.norm_k(txt_k)

        mask = rearrange(txt_mask, "b l -> b 1 1 l").expand(-1, 1, T * H * W, -1)
        attn = self.attn(vid_q, txt_k, txt_v, mask)

        vid_out = rearrange(attn, "b h (T H W) d -> b T H W (h d)", T=T, H=H, W=W)
        vid_out = gather_heads_scatter_seq(vid_out, head_dim=4, seq_dim=2)
        vid_out = self.proj_out(vid_out)

        return vid_out
