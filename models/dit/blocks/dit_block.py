from typing import Optional, Tuple
import torch
from torch import nn

from ..mlp import get_mlp
from ..modulation import ada_layer_type
from ..normalization import norm_layer_type
from .attention import get_attn


class TransformerBlock(nn.Module):
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
        vid: torch.FloatTensor,
        txt: Optional[torch.FloatTensor],
        txt_mask: Optional[torch.BoolTensor],
        emb: torch.FloatTensor,
    ) -> Tuple[
        torch.FloatTensor,
        torch.FloatTensor,
    ]:
        vid_msa = self.ada(vid, emb=emb, layer="msa", mode="in", norm_layer=self.msa_norm)
        vid_msa = self.msa(vid_msa)
        vid = self.ada(vid_msa, emb=emb, layer="msa", mode="out", residual=vid)

        if hasattr(self, "mca"):
            vid_mca = self.ada(vid, emb=emb, layer="mca", mode="in", norm_layer=self.mca_norm)
            vid_mca = self.mca(vid_mca, txt, txt_mask)
            vid = self.ada(vid_mca, emb=emb, layer="mca", mode="out", residual=vid)

        vid_mlp = self.ada(vid, emb=emb, layer="mlp", mode="in", norm_layer=self.mlp_norm)
        vid_mlp = self.mlp(vid_mlp)
        vid = self.ada(vid_mlp, emb=emb, layer="mlp", mode="out", residual=vid)

        return vid, txt
