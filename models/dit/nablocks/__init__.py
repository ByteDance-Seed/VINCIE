# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates  
# SPDX-License-Identifier: Apache-2.0
from .mmdit_block import NaMMTransformerBlock

nadit_blocks = {
    "mmdit": NaMMTransformerBlock,
}


def get_nablock(block_type: str):
    if block_type in nadit_blocks:
        return nadit_blocks[block_type]
    raise NotImplementedError(f"{block_type} is not supported")
