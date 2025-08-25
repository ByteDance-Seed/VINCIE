 # Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
 # SPDX-License-Identifier: Apache-2.0 

from contextlib import nullcontext
from typing import Callable, Union
from torch import nn
from torch.utils.checkpoint import checkpoint

from .contexts import RecomputingCtx
from .states import is_mlao_initialized


def gen_recompute_indicator_context_fn():
    if is_mlao_initialized():
        return RecomputingCtx(False), RecomputingCtx(True)
    return nullcontext(), nullcontext()


def gradient_checkpointing(module: Union[Callable, nn.Module], *args, enabled: bool, **kwargs):
    if enabled:
        return checkpoint(
            module,
            *args,
            use_reentrant=False,
            context_fn=gen_recompute_indicator_context_fn,
            **kwargs,
        )
    else:
        return module(*args, **kwargs)
