from typing import Tuple
import torch
from torch import nn

from ..mlp import get_mlp
from ..mm import MMArg, MMModule
from ..modulation import ada_layer_type
from ..normalization import norm_layer_type
from .attention import get_attn


class MMTransformerBlock(nn.Module):
    def __init__(
        self,
        *,
        vid_dim: int,
        txt_dim: int,
        emb_dim: int,
        heads: int,
        head_dim: int,
        expand_ratio: int,
        norm: norm_layer_type,
        norm_eps: float,
        ada: ada_layer_type,
        qk_bias: bool,
        qk_norm: norm_layer_type,
        shared_weights: bool,
        mlp_type: str,
        msa_type: str,
        rope_type: str,
        rope_dim: int,
        is_last_layer: bool,
        **kwargs,
    ):
        super().__init__()
        dim = MMArg(vid_dim, txt_dim)
        self.attn_norm = MMModule(
            norm,
            dim=dim,
            eps=norm_eps,
            elementwise_affine=False,
            shared_weights=shared_weights,
        )
        self.attn = get_attn(msa_type)(
            vid_dim=vid_dim,
            txt_dim=txt_dim,
            heads=heads,
            head_dim=head_dim,
            qk_bias=qk_bias,
            qk_norm=qk_norm,
            qk_norm_eps=norm_eps,
            rope_type=rope_type,
            rope_dim=rope_dim,
            shared_weights=shared_weights,
            window=kwargs.pop("window", None),
            window_method=kwargs.pop("window_method", None),
        )
        self.mlp_norm = MMModule(
            norm,
            dim=dim,
            eps=norm_eps,
            elementwise_affine=False,
            shared_weights=shared_weights,
            vid_only=is_last_layer,
        )
        self.mlp = MMModule(
            get_mlp(mlp_type),
            dim=dim,
            expand_ratio=expand_ratio,
            shared_weights=shared_weights,
            vid_only=is_last_layer,
        )
        self.ada = MMModule(
            ada,
            dim=dim,
            emb_dim=emb_dim,
            layers=["attn", "mlp"],
            shared_weights=shared_weights,
            vid_only=is_last_layer,
        )
        self.is_last_layer = is_last_layer

    def forward(
        self,
        vid: torch.FloatTensor,
        txt: torch.FloatTensor,
        txt_mask: torch.BoolTensor,
        emb: torch.FloatTensor,
    ) -> Tuple[
        torch.FloatTensor,
        torch.FloatTensor,
    ]:
        residual = MMArg(vid, txt)
        norm_layer = MMArg(self.attn_norm.get_module("vid"), self.attn_norm.get_module("txt"))
        vid_attn, txt_attn = self.ada(
            vid, txt, emb=emb, layer="attn", mode="in", norm_layer=norm_layer
        )
        vid_attn, txt_attn = self.attn(vid_attn, txt_attn, txt_mask=txt_mask)
        vid_attn, txt_attn = self.ada(
            vid_attn, txt_attn, emb=emb, layer="attn", mode="out", residual=residual
        )

        residual = MMArg(vid_attn, None if self.is_last_layer else txt_attn)
        norm_layer = MMArg(self.mlp_norm.get_module("vid"), self.mlp_norm.get_module("txt"))
        vid_mlp, txt_mlp = self.ada(
            vid_attn, txt_attn, emb=emb, layer="mlp", mode="in", norm_layer=norm_layer
        )
        vid_mlp, txt_mlp = self.mlp(vid_mlp, txt_mlp)
        vid_mlp, txt_mlp = self.ada(
            vid_mlp, txt_mlp, emb=emb, layer="mlp", mode="out", residual=residual
        )

        return vid_mlp, txt_mlp
