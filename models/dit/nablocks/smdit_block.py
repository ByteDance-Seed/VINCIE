from typing import Tuple
import torch
from torch import nn

from common.cache_v2 import Cache

from ..mlp import get_mlp
from ..mm import MMArg, MMModule
from ..modulation import ada_layer_type
from ..normalization import norm_layer_type
from .attention import get_attn


class NaSMTransformerBlock(nn.Module):
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
        mca_type: str,
        rope_type: str,
        rope_dim: int,
        **kwargs,
    ):
        super().__init__()
        dim = MMArg(vid_dim, txt_dim)
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
            self.mca_norm = MMModule(
                norm,
                dim=vid_dim,
                eps=norm_eps,
                elementwise_affine=False,
                shared_weights=shared_weights,
            )
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
                shared_weights=shared_weights,
            )

        self.mlp_norm = MMModule(
            norm, dim=dim, eps=norm_eps, elementwise_affine=False, shared_weights=shared_weights
        )
        self.mlp = MMModule(
            get_mlp(mlp_type),
            dim=dim,
            expand_ratio=expand_ratio,
            shared_weights=shared_weights,
        )

        self.ada = MMModule(
            ada,
            dim=dim,
            emb_dim=emb_dim,
            layers=["msa", "mca", "mlp"] if txt_dim else ["msa", "mlp"],
            shared_weights=shared_weights,
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
        ada_msa_kwargs = {
            "emb": emb,
            "hid_shape": vid_shape,
            "cache": cache,
            "branch_tag": "vid",
        }

        ada_kwargs = {
            "emb": emb,
            "hid_shape": MMArg(vid_shape, txt_shape),
            "cache": cache,
            "branch_tag": MMArg("vid", "txt"),
        }

        vid_msa = self.ada.vid(
            vid, layer="msa", mode="in", norm_layer=self.msa_norm, **ada_msa_kwargs
        )
        vid_msa = self.msa(vid_msa, vid_shape, cache)
        vid = self.ada.vid(vid_msa, layer="msa", mode="out", residual=vid, **ada_msa_kwargs)

        if hasattr(self, "mca"):
            residual = MMArg(vid, txt)
            norm_layer = MMArg(self.mca_norm.get_module("vid"), self.mca_norm.get_module("txt"))
            vid_mca, txt_mca = self.ada(
                vid, txt, layer="mca", mode="in", norm_layer=norm_layer, **ada_kwargs
            )
            vid_mca, txt_mca = self.mca(vid_mca, txt_mca, vid_shape, txt_shape, cache)
            vid, txt = self.ada(
                vid_mca, txt_mca, layer="mca", mode="out", residual=residual, **ada_kwargs
            )

        residual = MMArg(vid, txt)
        norm_layer = MMArg(self.mlp_norm.get_module("vid"), self.mlp_norm.get_module("txt"))
        vid_mlp, txt_mlp = self.ada(
            vid, txt, layer="mlp", mode="in", norm_layer=norm_layer, **ada_kwargs
        )
        vid_mlp, txt_mlp = self.mlp(vid_mlp, txt_mlp)
        vid, txt = self.ada(
            vid_mlp, txt_mlp, layer="mlp", mode="out", residual=residual, **ada_kwargs
        )

        return vid, txt, vid_shape, txt_shape
