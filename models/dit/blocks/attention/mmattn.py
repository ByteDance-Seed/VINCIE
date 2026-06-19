from typing import Optional, Tuple, Union
import torch
from einops import rearrange, repeat
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.utils import _triple

from common.distributed.ops import (
    gather_heads,
    gather_heads_scatter_seq,
    gather_seq_scatter_heads_qkv,
    scatter_heads,
)

from ...attention import TorchAttention
from ...mm import MMArg, MMModule
from ...normalization import norm_layer_type
from ...rope import get_rope
from ...window import make_shifted_windows


class MMAttention(nn.Module):
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
        self.rope = get_rope(rope_type=rope_type, dim=rope_dim)
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
        vid_qkv, txt_qkv = self.proj_qkv(vid, txt)
        vid_qkv = gather_seq_scatter_heads_qkv(vid_qkv, seq_dim=2)
        _, T, H, W, _ = vid_qkv.shape
        vid_qkv = rearrange(vid_qkv, "b T H W (o h d) -> o b h (T H W) d", o=3, d=self.head_dim)
        txt_qkv = rearrange(txt_qkv, "b L (o h d) -> o b h L d", o=3, d=self.head_dim)
        txt_qkv = scatter_heads(txt_qkv, dim=2)

        vid_len = vid_qkv.size(-2)
        txt_len = txt_qkv.size(-2)

        vid_q, vid_k, vid_v = vid_qkv.unbind()
        txt_q, txt_k, txt_v = txt_qkv.unbind()

        vid_q, txt_q = self.norm_q(vid_q, txt_q)
        vid_k, txt_k = self.norm_k(vid_k, txt_k)

        if self.rope:
            vid_q, vid_k = self.rope(vid_q, vid_k, (T, H, W))

        q = torch.cat([vid_q, txt_q], dim=-2)
        k = torch.cat([vid_k, txt_k], dim=-2)
        v = torch.cat([vid_v, txt_v], dim=-2)

        mask = F.pad(txt_mask, (vid_len, 0), value=True)
        mask = rearrange(mask, "b l -> b 1 1 l").expand(-1, 1, vid_len + txt_len, -1)
        attn = self.attn(q, k, v, mask)

        vid_out, txt_out = attn.split([vid_len, txt_len], dim=-2)

        vid_out = rearrange(vid_out, "b h (T H W) d -> b T H W (h d)", T=T, H=H, W=W)
        vid_out = gather_heads_scatter_seq(vid_out, head_dim=4, seq_dim=2)
        txt_out = rearrange(txt_out, "b h L d -> b L (h d)")
        txt_out = gather_heads(txt_out, dim=2)
        vid_out, txt_out = self.proj_out(vid_out, txt_out)

        return vid_out, txt_out


class MMSpaceAttention(MMAttention):
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

        # Add rotary embedding.
        if self.rope:
            vid_q, vid_k = self.rope(vid_q, vid_k, (T, H, W))

        # Define transformation function to 2d spatial attention.
        def vid_spatial(v):
            return rearrange(v, "b h (T H W) d -> b h T (H W) d", T=T, H=H, W=W)

        def txt_spatial(t):
            return rearrange(t, "b h L d -> b h 1 L d").expand(-1, -1, T, -1, -1)

        # Process video attention.
        vid_msk = F.pad(txt_mask, (H * W, 0), value=True)
        vid_msk = rearrange(vid_msk, "b l -> b 1 1 1 l").expand(-1, 1, 1, H * W, -1)
        vid_out = self.attn(
            vid_spatial(vid_q),
            torch.cat([vid_spatial(vid_k), txt_spatial(txt_k)], dim=-2),
            torch.cat([vid_spatial(vid_v), txt_spatial(txt_v)], dim=-2),
            vid_msk,
        )
        vid_out = rearrange(vid_out, "b h T (H W) d -> b T H W (h d)", H=H, W=W)
        vid_out = gather_heads_scatter_seq(vid_out, head_dim=4, seq_dim=2)

        # Process text attention.
        txt_msk = F.pad(txt_mask, (T * H * W, 0), value=True)
        txt_msk = rearrange(txt_msk, "b l -> b 1 1 l").expand(-1, 1, L, -1)
        txt_out = self.attn(
            txt_q,
            torch.cat([vid_k, txt_k], dim=-2),
            torch.cat([vid_v, txt_v], dim=-2),
            txt_msk,
        )
        txt_out = rearrange(txt_out, "b h L d -> b L (h d)")
        txt_out = gather_heads(txt_out, dim=2)

        # Project output.
        vid_out, txt_out = self.proj_out(vid_out, txt_out)
        return vid_out, txt_out


class MMWindowAttention(MMAttention):
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
        txt_qkv = rearrange(txt_qkv, "b L (o h d) -> o b h L d", o=3, d=self.head_dim)
        txt_qkv = scatter_heads(txt_qkv, dim=2)

        vid_q, vid_k, vid_v = vid_qkv.unbind()
        txt_q, txt_k, txt_v = txt_qkv.unbind()

        vid_q, txt_q = self.norm_q(vid_q, txt_q)
        vid_k, txt_k = self.norm_k(vid_k, txt_k)

        if self.rope:
            vid_q, vid_k = self.rope(vid_q, vid_k, (T, H, W))

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

        def txt_window(t):
            return rearrange(t, "b h L d -> b h 1 L d").expand(-1, -1, nt * nh * nw, -1, -1)

        # Process video attention.
        vid_msk = F.pad(txt_mask, (tt * hh * ww, 0), value=True)
        vid_msk = rearrange(vid_msk, "b l -> b 1 1 1 l").expand(-1, 1, 1, tt * hh * ww, -1)
        vid_out = self.attn(
            vid_window(vid_q),
            torch.cat([vid_window(vid_k), txt_window(txt_k)], dim=-2),
            torch.cat([vid_window(vid_v), txt_window(txt_v)], dim=-2),
            vid_msk,
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

        # Process text attention.
        txt_msk = F.pad(txt_mask, (T * H * W, 0), value=True)
        txt_msk = rearrange(txt_msk, "b l -> b 1 1 l").expand(-1, 1, L, -1)
        txt_out = self.attn(
            txt_q,
            torch.cat([vid_k, txt_k], dim=-2),
            torch.cat([vid_v, txt_v], dim=-2),
            txt_msk,
        )
        txt_out = rearrange(txt_out, "b h L d -> b L (h d)")
        txt_out = gather_heads(txt_out, dim=2)

        # Project output.
        vid_out, txt_out = self.proj_out(vid_out, txt_out)
        return vid_out, txt_out


class MMShiftedWindowAttention(nn.Module):
    def __init__(
        self,
        vid_dim: int,
        txt_dim: int,
        heads: int,
        head_dim: int,
        qk_bias: bool,
        qk_norm: norm_layer_type,
        qk_norm_eps: float,
        window: Union[int, Tuple[int, int, int]],
        window_method: str,
        rope_type: Optional[str],
        rope_dim: int,
        shared_weights: bool,
    ):
        super().__init__()
        dim = MMArg(vid_dim, txt_dim)
        inner_dim = heads * head_dim
        qkv_dim = inner_dim * 3

        self.window = _triple(window)
        self.window_method = window_method
        assert all(map(lambda v: isinstance(v, int) and v >= 0, self.window))

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
        self.rope = get_rope(rope_type=rope_type, dim=rope_dim)
        self.attn = TorchAttention()

    def get_shift_attention_mask(
        self,
        size: tuple[int, int, int],
        window: tuple[int, int, int],
        shift_window: tuple[int, int, int],
    ):
        nt, nh, nw = window
        st, sh, sw = shift_window
        window_slices = make_shifted_windows(size, window)
        img_mask = torch.zeros(size)

        cnt = 0
        for it, ih, iw in window_slices:
            img_mask[it, ih, iw] = cnt
            cnt += 1

        img_mask = torch.roll(img_mask, (-st, -sh, -sw), (0, 1, 2))
        mask_windows = rearrange(
            img_mask,
            "(nt tt) (nh hh) (nw ww) -> (nt nh nw) (tt hh ww)",
            nt=nt,
            nh=nh,
            nw=nw,
        )
        attn_mask = mask_windows.unsqueeze(1) == mask_windows.unsqueeze(2)
        return attn_mask

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
        B, T, H, W, _ = vid_qkv.shape
        _, L, _ = txt.shape

        if self.window_method == "swin":
            nt, nh, nw = self.window
            tt, hh, ww = T // nt, H // nh, W // nw
        elif self.window_method == "swin_by_size":
            tt, hh, ww = self.window
            tt, hh, ww = (
                tt if tt > 0 else T,
                hh if hh > 0 else H,
                ww if ww > 0 else W,
            )
            nt, nh, nw = T // tt, H // hh, W // ww
        else:
            raise NotImplementedError

        st, sh, sw = tt // 2, hh // 2, ww // 2

        vid_qkv = rearrange(vid_qkv, "b T H W (o h d) -> o b h (T H W) d", o=3, d=self.head_dim)
        txt_qkv = rearrange(txt_qkv, "b L (o h d) -> o b h L d", o=3, d=self.head_dim)
        txt_qkv = scatter_heads(txt_qkv, dim=2)

        vid_q, vid_k, vid_v = vid_qkv.unbind()
        txt_q, txt_k, txt_v = txt_qkv.unbind()

        vid_q, txt_q = self.norm_q(vid_q, txt_q)
        vid_k, txt_k = self.norm_k(vid_k, txt_k)

        if self.rope:
            vid_q, vid_k = self.rope(vid_q, vid_k, (T, H, W))

        def vid_window(v):
            v = rearrange(v, "b h (T H W) d -> b h T H W d", H=H, W=W)
            v = torch.roll(v, (-st, -sh, -sw), (2, 3, 4))
            return rearrange(
                v,
                "b h (nt tt) (nh hh) (nw ww) d -> (b nt nh nw) h (tt hh ww) d",
                tt=tt,
                hh=hh,
                ww=ww,
                nt=nt,
                nh=nh,
                nw=nw,
            )

        def txt_window(t):
            return repeat(t, "b h L d -> (b nt nh nw) h L d", nt=nt, nh=nh, nw=nw)

        # Process video attention.
        vid_msk = self.get_shift_attention_mask((T, H, W), (nt, nh, nw), (st, sh, sw)).to(
            txt_mask.device
        )
        vid_msk = repeat(vid_msk, "n l1 l2 -> (b n) 1 l1 l2", b=B)
        txt_msk = repeat(txt_mask, "b tl -> (b n) 1 vl tl", n=nt * nh * nw, vl=tt * hh * ww)

        vid_msk = torch.cat([vid_msk, txt_msk], dim=-1)

        vid_out = self.attn(
            vid_window(vid_q),
            torch.cat([vid_window(vid_k), txt_window(txt_k)], dim=-2),
            torch.cat([vid_window(vid_v), txt_window(txt_v)], dim=-2),
            vid_msk,
        )
        vid_out = rearrange(
            vid_out,
            "(b nt nh nw) h (tt hh ww) d -> b (nt tt) (nh hh) (nw ww) (h d)",
            tt=tt,
            hh=hh,
            ww=ww,
            nt=nt,
            nh=nh,
            nw=nw,
        )
        vid_out = gather_heads_scatter_seq(vid_out, head_dim=4, seq_dim=2)
        vid_out = torch.roll(vid_out, (st, sh, sw), (1, 2, 3))

        # Process text attention.
        txt_msk = F.pad(txt_mask, (T * H * W, 0), value=True)
        txt_msk = rearrange(txt_msk, "b l -> b 1 1 l").expand(-1, 1, L, -1)
        txt_out = self.attn(
            txt_q,
            torch.cat([vid_k, txt_k], dim=-2),
            torch.cat([vid_v, txt_v], dim=-2),
            txt_msk,
        )
        txt_out = rearrange(txt_out, "b h L d -> b L (h d)")
        txt_out = gather_heads(txt_out, dim=2)

        # Project output.
        vid_out, txt_out = self.proj_out(vid_out, txt_out)
        return vid_out, txt_out
