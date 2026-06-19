from typing import Optional, Tuple, Union
import torch
from einops import rearrange
from torch import nn
from torch.nn.modules.utils import _triple

from common.distributed.ops import gather_heads_scatter_seq, gather_seq_scatter_heads_qkv

from ...attention import TorchAttention
from ...normalization import norm_layer_type
from ...rope import get_rope


class SelfAttention(nn.Module):
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
        self.rope = get_rope(rope_type=rope_type, dim=rope_dim)
        self.attn = TorchAttention()

    def forward(
        self,
        vid: torch.FloatTensor,  # b T H W c
    ) -> torch.FloatTensor:

        vid_qkv = self.proj_qkv(vid)
        vid_qkv = gather_seq_scatter_heads_qkv(vid_qkv, seq_dim=2)
        _, T, H, W, _ = vid_qkv.shape
        vid_qkv = rearrange(vid_qkv, "b T H W (o h d) -> o b h (T H W) d", o=3, d=self.head_dim)

        vid_q, vid_k, vid_v = vid_qkv.unbind()

        vid_q = self.norm_q(vid_q)
        vid_k = self.norm_k(vid_k)

        if self.rope:
            vid_q, vid_k = self.rope(vid_q, vid_k, (T, H, W))

        vid_out = self.attn(vid_q, vid_k, vid_v)
        vid_out = rearrange(vid_out, "b h (T H W) d -> b T H W (h d)", T=T, H=H, W=W)
        vid_out = gather_heads_scatter_seq(vid_out, head_dim=4, seq_dim=2)
        vid_out = self.proj_out(vid_out)

        return vid_out


class SpaceAttention(SelfAttention):
    def forward(
        self,
        vid: torch.FloatTensor,  # b T H W c
    ) -> torch.FloatTensor:

        vid_qkv = self.proj_qkv(vid)
        vid_qkv = gather_seq_scatter_heads_qkv(vid_qkv, seq_dim=2)
        _, T, H, W, _ = vid_qkv.shape
        vid_qkv = rearrange(vid_qkv, "b T H W (o h d) -> o b h (T H W) d", o=3, d=self.head_dim)

        vid_q, vid_k, vid_v = vid_qkv.unbind()

        vid_q = self.norm_q(vid_q)
        vid_k = self.norm_k(vid_k)

        if self.rope:
            vid_q, vid_k = self.rope(vid_q, vid_k, (T, H, W))

        def vid_spatial(v):
            return rearrange(v, "b h (T H W) d -> (b T) h (H W) d", T=T, H=H, W=W)

        vid_out = self.attn(vid_spatial(vid_q), vid_spatial(vid_k), vid_spatial(vid_v))
        vid_out = rearrange(vid_out, "(b T) h (H W) d -> b T H W (h d)", T=T, H=H, W=W)
        vid_out = gather_heads_scatter_seq(vid_out, head_dim=4, seq_dim=2)
        vid_out = self.proj_out(vid_out)

        return vid_out


class TemporalAttention(SelfAttention):
    def forward(
        self,
        vid: torch.FloatTensor,  # b T H W c
    ) -> torch.FloatTensor:

        vid_qkv = self.proj_qkv(vid)
        vid_qkv = gather_seq_scatter_heads_qkv(vid_qkv, seq_dim=2)
        _, T, H, W, _ = vid_qkv.shape
        vid_qkv = rearrange(vid_qkv, "b T H W (o h d) -> o b h (T H W) d", o=3, d=self.head_dim)

        vid_q, vid_k, vid_v = vid_qkv.unbind()

        vid_q = self.norm_q(vid_q)
        vid_k = self.norm_k(vid_k)

        if self.rope:
            vid_q, vid_k = self.rope(vid_q, vid_k, (T, H, W))

        def vid_temporal(v):
            return rearrange(v, "b h (T H W) d -> (b H W) h T d", T=T, H=H, W=W)

        vid_out = self.attn(vid_temporal(vid_q), vid_temporal(vid_k), vid_temporal(vid_v))
        vid_out = rearrange(vid_out, "(b H W) h T d -> b T H W (h d)", T=T, H=H, W=W)
        vid_out = gather_heads_scatter_seq(vid_out, head_dim=4, seq_dim=2)
        vid_out = self.proj_out(vid_out)

        return vid_out


class WindowAttention(SelfAttention):
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

    def forward(
        self,
        vid: torch.FloatTensor,  # b T H W c
    ) -> Tuple[torch.FloatTensor,]:
        # Project q, k, v.
        vid_qkv = self.proj_qkv(vid)
        vid_qkv = gather_seq_scatter_heads_qkv(vid_qkv, seq_dim=2)
        _, T, H, W, _ = vid_qkv.shape

        if self.window_method == "win":
            nt, nh, nw = self.window
            tt, hh, ww = T // nt, H // nh, W // nw
        elif self.window_method == "win_by_size":
            tt, hh, ww = self.window
            tt, hh, ww = (
                tt if tt > 0 else T,
                hh if hh > 0 else H,
                ww if ww > 0 else W,
            )
            nt, nh, nw = T // tt, H // hh, W // ww
        else:
            raise NotImplementedError

        vid_qkv = rearrange(vid_qkv, "b T H W (o h d) -> o b h (T H W) d", o=3, d=self.head_dim)
        vid_q, vid_k, vid_v = vid_qkv.unbind()

        vid_q, vid_k = self.norm_q(vid_q), self.norm_k(vid_k)

        # Add rotary embedding.
        if self.rope:
            vid_q, vid_k = self.rope(vid_q, vid_k, (T, H, W))

        # Define transformation function to 2d spatial attention.
        def vid_window(v):
            return rearrange(
                v,
                "b h (nt tt nh hh nw ww) d -> b h (nt nh nw) (tt hh ww) d",
                hh=hh,
                ww=ww,
                tt=tt,
                nh=nh,
                nw=nw,
                nt=nt,
            )

        # Process video attention.
        vid_out = self.attn(
            vid_window(vid_q),
            vid_window(vid_k),
            vid_window(vid_v),
        )
        vid_out = rearrange(
            vid_out,
            "b h (nt nh nw) (tt hh ww) d -> b (nt tt) (nh hh) (nw ww) (h d)",
            hh=hh,
            ww=ww,
            tt=tt,
            nh=nh,
            nw=nw,
        )
        vid_out = gather_heads_scatter_seq(vid_out, head_dim=4, seq_dim=2)

        # Project output.
        vid_out = self.proj_out(vid_out)
        return vid_out
