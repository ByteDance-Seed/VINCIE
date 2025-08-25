# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates  
# SPDX-License-Identifier: Apache-2.0
from typing import List, Tuple
from torch import nn
from torch.utils.checkpoint import checkpoint

from .types import _memory_list_t, _tensor_t


class CausalSequential(nn.ModuleList):
    def __init__(
        self,
        modules: List[nn.Module],
        checkpointing: bool = False,
    ):
        super().__init__(modules)
        self.checkpointing = checkpointing

    def forward(
        self,
        x: _tensor_t,
        m: _memory_list_t,
    ) -> Tuple[
        _tensor_t,
        _memory_list_t,
    ]:
        m = m or [None] * len(self)
        for i, module in enumerate(self):
            if self.checkpointing and self.training:
                x, m[i] = checkpoint(module, x, m[i], use_reentrant=False)
            else:
                x, m[i] = module(x, m[i])
        return x, m
