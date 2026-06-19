from .attn import SelfAttention, SpaceAttention, TemporalAttention, WindowAttention
from .cross import CrossAttention
from .mmattn import MMAttention, MMShiftedWindowAttention, MMSpaceAttention, MMWindowAttention
from .mmcross import MMCrossAttention

attns = {
    "cross": CrossAttention,
    "full": SelfAttention,
    "space": SpaceAttention,
    "temporal": TemporalAttention,
    "window": WindowAttention,
    "mm_cross": MMCrossAttention,
    "mm_full": MMAttention,
    "mm_shifted_window": MMShiftedWindowAttention,
    "mm_space": MMSpaceAttention,
    "mm_window": MMWindowAttention,
}


def get_attn(attn_type: str):
    if attn_type in attns:
        return attns[attn_type]
    raise NotImplementedError(f"{attn_type} is not supported")
