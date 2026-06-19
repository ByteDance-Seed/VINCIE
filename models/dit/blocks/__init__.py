from .dit_block import TransformerBlock
from .mmdit_block import MMTransformerBlock
from .smdit_block import SMTransformerBlock

dit_blocks = {
    "dit": TransformerBlock,
    "mmdit": MMTransformerBlock,
    "smdit": SMTransformerBlock,
}


def get_block(block_type: str):
    if block_type in dit_blocks:
        return dit_blocks[block_type]
    raise NotImplementedError(f"{block_type} is not supported")
