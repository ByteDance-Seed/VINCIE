from typing import Tuple
import torch
from einops import rearrange, repeat
from torch import nn

from common.distributed.ops import (
    gather_heads,
    gather_heads_scatter_seq,
    gather_seq_scatter_heads_qkv,
    scatter_heads,
)

from ...attention import TorchAttention
from ...mm import MMArg, MMModule
from ...normalization import norm_layer_type


class MMCrossAttention(nn.Module):
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
        self.attn = TorchAttention()

    def forward(
        self,
        vid: torch.FloatTensor,  # b T H W c
        txt: torch.FloatTensor,  # b L c
        txt_mask: torch.BoolTensor,  # b L
    ) -> Tuple[
        torch.FloatTensor,
        torch.FloatTensor,
    ]:
        # Project q, k, v.
        vid_qkv, txt_qkv = self.proj_qkv(vid, txt)
        vid_qkv = gather_seq_scatter_heads_qkv(vid_qkv, seq_dim=2)
        _, T, H, W, _ = vid_qkv.shape
        _, L, _ = txt.shape
        vid_qkv = rearrange(vid_qkv, "b T H W (o h d) -> o b h (T H W) d", o=3, d=self.head_dim)
        txt_qkv = rearrange(txt_qkv, "b L (o h d) -> o b h L d", o=3, d=self.head_dim)
        txt_qkv = scatter_heads(txt_qkv, dim=2)

        # Split to q, k, v.
        vid_q, vid_k, vid_v = vid_qkv.unbind()
        txt_q, txt_k, txt_v = txt_qkv.unbind()

        # Normalize q, k.
        vid_q, txt_q = self.norm_q(vid_q, txt_q)
        vid_k, txt_k = self.norm_k(vid_k, txt_k)

        # Process video attention.
        attn_mask = repeat(txt_mask, "b tl -> b 1 vl tl", vl=T * H * W)
        vid_out = self.attn(
            vid_q,
            txt_k,
            txt_v,
            attn_mask=attn_mask,
        )
        vid_out = rearrange(vid_out, "b h (T H W) d -> b T H W (h d)", H=H, W=W)
        vid_out = gather_heads_scatter_seq(vid_out, head_dim=4, seq_dim=2)

        # Process text attention.
        attn_mask = repeat(txt_mask, "b tl -> b 1 tl l", l=L + T * H * W)
        txt_out = self.attn(
            txt_q,
            torch.cat([vid_k, txt_k], dim=-2),
            torch.cat([vid_v, txt_v], dim=-2),
            attn_mask=attn_mask,
        )
        txt_out = rearrange(txt_out, "b h L d -> b L (h d)")
        txt_out = gather_heads(txt_out, dim=2)

        # Project output.
        vid_out, txt_out = self.proj_out(vid_out, txt_out)
        return vid_out, txt_out
