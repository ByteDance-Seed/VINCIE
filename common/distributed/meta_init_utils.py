 # Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
 # SPDX-License-Identifier: Apache-2.0 

import torch
from rotary_embedding_torch import RotaryEmbedding
from torch import nn
from torch.distributed.fsdp._common_utils import _is_fsdp_flattened

__all__ = ["meta_param_init_fn", "meta_non_persistent_buffer_init_fn"]


def meta_param_init_fn(module: nn.Module) -> None:
    """
    Used for model inited onto meta device.
    Init meta param/buffer with empty tensor.
    We don't care numerical correctness in this func.
    FSDP will sync param/buffer state from rank0 to the other ranks.
    """

    with torch.no_grad():
        for submodule in module.modules():
            for param_name, param in submodule.named_parameters(recurse=False):
                if not _is_fsdp_flattened(param) and param.is_meta:
                    materialized_param = nn.Parameter(torch.empty_like(param, device="cpu"))
                    setattr(submodule, param_name, materialized_param)
            for buffer_name, buffer in submodule.named_buffers(recurse=False):
                if not _is_fsdp_flattened(buffer) and buffer.is_meta:
                    materialized_param = torch.empty_like(buffer, device="cpu")
                    setattr(submodule, buffer_name, materialized_param)
            torch.cuda.empty_cache()


def meta_non_persistent_buffer_init_fn(module: nn.Module) -> nn.Module:
    """
    Used for materializing `non-persistent tensor buffers` while model resuming.

    Since non-persistent tensor buffers are not saved in state_dict,
    when initializing model with meta device, user should materialize those buffers manually.

    Currently, only `rope.dummy` is this special case.
    """
    with torch.no_grad():
        for submodule in module.modules():
            if not isinstance(submodule, RotaryEmbedding):
                continue
            for buffer_name, buffer in submodule.named_buffers(recurse=False):
                if buffer.is_meta and "dummy" in buffer_name:
                    materialized_buffer = torch.zeros_like(buffer, device="cpu")
                    setattr(submodule, buffer_name, materialized_buffer)
    assert not any(b.is_meta for n, b in module.named_buffers())
    return module
