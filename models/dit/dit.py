from typing import Optional, Tuple, Union
import torch
from torch import nn

from common.mlao import gradient_checkpointing

from .blocks import get_block
from .embedding import TimeEmbedding, emb_add
from .modulation import get_ada_layer
from .normalization import get_norm_layer
from .patch import get_patch_layers


class DiT(nn.Module):
    """
    Diffusion Transformer (DiT)
    """

    gradient_checkpointing = False

    def __init__(
        self,
        *,
        vid_in_channels: int,
        vid_out_channels: int,
        vid_dim: int,
        txt_in_dim: Optional[int] = None,
        txt_dim: Optional[int],
        emb_dim: int,
        heads: int,
        head_dim: int,
        expand_ratio: int,
        norm: Optional[str],
        norm_eps: float = 1e-5,
        ada: str,
        qk_bias: bool = False,
        qk_norm: Optional[str],
        patch_size: Union[int, Tuple[int, int, int]],
        num_layers: int,
        block_type: Union[str, Tuple[str]],
        mm_layers: Union[int, Tuple[bool]],
        mlp_type: str = "normal",
        rope_type: Optional[str] = "rope3d",
        rope_dim: Optional[int] = None,
        window: Optional[Tuple] = None,
        window_method: Optional[Tuple[str]] = None,
        msa_type: Optional[Tuple[str]] = None,
        mca_type: Optional[Tuple[str]] = None,
        patch_type: str = "v1",
        **kwargs,
    ):
        ada = get_ada_layer(ada, norm)
        norm = get_norm_layer(norm)
        qk_norm = get_norm_layer(qk_norm)
        rope_dim = rope_dim if rope_dim is not None else head_dim // 2
        if isinstance(block_type, str):
            block_type = [block_type] * num_layers
        elif len(block_type) != num_layers:
            raise ValueError("The ``block_type`` list should equal to ``num_layers``.")
        super().__init__()
        PatchIn, PatchOut = get_patch_layers(patch_type)
        self.vid_in = PatchIn(
            in_channels=vid_in_channels,
            patch_size=patch_size,
            dim=vid_dim,
        )
        self.txt_in = (
            nn.Linear(txt_in_dim, txt_dim)
            if txt_in_dim and txt_in_dim != txt_dim
            else nn.Identity()
        )
        self.emb_in = TimeEmbedding(
            sinusoidal_dim=256,
            hidden_dim=max(vid_dim, txt_dim or 0),
            output_dim=emb_dim,
        )

        if window is None or isinstance(window[0], int):
            window = [window] * num_layers
        if window_method is None or isinstance(window_method, str):
            window_method = [window_method] * num_layers

        if msa_type is None or isinstance(msa_type, str):
            msa_type = [msa_type] * num_layers
        if mca_type is None or isinstance(mca_type, str):
            mca_type = [mca_type] * num_layers

        self.blocks = nn.ModuleList(
            [
                get_block(block_type[i])(
                    vid_dim=vid_dim,
                    txt_dim=txt_dim,
                    emb_dim=emb_dim,
                    heads=heads,
                    head_dim=head_dim,
                    expand_ratio=expand_ratio,
                    norm=norm,
                    norm_eps=norm_eps,
                    ada=ada,
                    qk_bias=qk_bias,
                    qk_norm=qk_norm,
                    shared_weights=not (
                        (i < mm_layers) if isinstance(mm_layers, int) else mm_layers[i]
                    ),
                    mlp_type=mlp_type,
                    window=window[i],
                    window_method=window_method[i],
                    msa_type=msa_type[i],
                    mca_type=mca_type[i],
                    rope_type=rope_type,
                    rope_dim=rope_dim,
                    is_last_layer=(i == num_layers - 1),
                    **kwargs,
                )
                for i in range(num_layers)
            ]
        )
        self.vid_out = PatchOut(
            out_channels=vid_out_channels,
            patch_size=patch_size,
            dim=vid_dim,
        )

    def set_gradient_checkpointing(self, enable: bool):
        self.gradient_checkpointing = enable

    def forward(
        self,
        *,
        vid: torch.FloatTensor,  # b c t h w
        txt: Optional[torch.FloatTensor] = None,  # b l d
        txt_mask: Optional[torch.BoolTensor] = None,  # b l
        emb: Optional[torch.FloatTensor] = None,  # b d
        timestep: Union[int, float, torch.IntTensor, torch.FloatTensor],  # b
    ):
        # Preprocessing
        if txt_mask is None and txt is not None:
            txt_mask = torch.full(txt.shape[:2], fill_value=True, device=txt.device)

        # Input
        vid = self.vid_in(vid)
        txt = self.txt_in(txt)
        emb = emb_add(self.emb_in(timestep, device=vid.device, dtype=vid.dtype), emb)

        # Body
        for block in self.blocks:
            vid, txt = gradient_checkpointing(
                enabled=(self.gradient_checkpointing and self.training),
                module=block,
                vid=vid,
                txt=txt,
                txt_mask=txt_mask,
                emb=emb,
            )

        # Output
        vid = self.vid_out(vid)
        return vid
