# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates  
# SPDX-License-Identifier: Apache-2.0
from torch import nn

from .types import _activation_t


def get_act_layer(activation: _activation_t) -> nn.Module:
    if activation is None:
        return nn.Identity()
    if activation == "silu":
        return nn.SiLU()
    raise NotImplementedError
