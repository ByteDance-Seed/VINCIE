from typing import Optional, Tuple
import torch
from torch import nn

from common.cache_v2 import Cache

from ..mlp import get_mlp
from ..modulation import ada_layer_type
from ..normalization import norm_layer_type
from .attention import get_attn


class NaTransformerBlock(nn.Module):
    def __init__(
        self,
        *,
        vid_dim: int,
        txt_dim: Optional[int],
        emb_dim: int,
        heads: int,
        head_dim: int,
        expand_ratio: int,
        norm: norm_layer_type,
        norm_eps: float,
        ada: ada_layer_type,
        qk_bias: bool,
        qk_norm: norm_layer_type,
        mlp_type: str,
        msa_type: str,
        mca_type: str,
        rope_type: str,
        rope_dim: int,
        **kwargs,
    ):
        super().__init__()
        self.msa_norm = norm(dim=vid_dim, eps=norm_eps, elementwise_affine=False)
        self.msa = get_attn(msa_type)(
            vid_dim=vid_dim,
            heads=heads,
            head_dim=head_dim,
            qk_bias=qk_bias,
            qk_norm=qk_norm,
            qk_norm_eps=norm_eps,
            rope_type=rope_type,
            rope_dim=rope_dim,
            window=kwargs.pop("window", None),
            window_method=kwargs.pop("window_method", None),
        )

        if txt_dim:
            self.mca_norm = norm(dim=vid_dim, eps=norm_eps, elementwise_affine=False)
            self.mca = get_attn(mca_type)(
                vid_dim=vid_dim,
                txt_dim=txt_dim,
                heads=heads,
                head_dim=head_dim,
                qk_bias=qk_bias,
                qk_norm=qk_norm,
                qk_norm_eps=norm_eps,
                rope_type=rope_type,
                rope_dim=rope_dim,
            )

        self.mlp_norm = norm(dim=vid_dim, eps=norm_eps, elementwise_affine=False)
        mlp_func = get_mlp(mlp_type)
        self.mlp = mlp_func(dim=vid_dim, expand_ratio=expand_ratio)
        self.ada = ada(
            dim=vid_dim,
            emb_dim=emb_dim,
            layers=["msa", "mca", "mlp"] if txt_dim else ["msa", "mlp"],
        )

    def forward(
        self,
        vid: torch.FloatTensor,  # l c
        txt: torch.FloatTensor,  # l c
        vid_shape: torch.LongTensor,  # b 3
        txt_shape: torch.LongTensor,  # b 1
        emb: torch.FloatTensor,
        cache: Cache,
    ) -> Tuple[
        torch.FloatTensor,
        torch.FloatTensor,
        torch.LongTensor,
        torch.LongTensor,
    ]:

        ada_kwargs = {
            "hid_shape": vid_shape,
            "cache": cache,
            "branch_tag": "vid",
        }
        vid_msa = self.ada(
            vid, emb=emb, layer="msa", mode="in", norm_layer=self.msa_norm, **ada_kwargs
        )
        vid_msa = self.msa(vid_msa, vid_shape, cache)
        vid = self.ada(vid_msa, emb=emb, layer="msa", mode="out", residual=vid, **ada_kwargs)

        if hasattr(self, "mca"):
            vid_mca = self.ada(
                vid, emb=emb, layer="mca", mode="in", norm_layer=self.mca_norm, **ada_kwargs
            )
            vid_mca = self.mca(vid_mca, txt, vid_shape, txt_shape, cache)
            vid = self.ada(vid_mca, emb=emb, layer="mca", mode="out", residual=vid, **ada_kwargs)

        vid_mlp = self.ada(
            vid, emb=emb, layer="mlp", mode="in", norm_layer=self.mlp_norm, **ada_kwargs
        )
        vid_mlp = self.mlp(vid_mlp)
        vid = self.ada(vid_mlp, emb=emb, layer="mlp", mode="out", residual=vid, **ada_kwargs)

        return vid, txt, vid_shape, txt_shape
