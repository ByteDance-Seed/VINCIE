 # Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
 # SPDX-License-Identifier: Apache-2.0 

"""
Utility functions for saving distributed states.
"""

from typing import Dict
import torch
from omegaconf import ListConfig
from torch._dynamo.eval_frame import OptimizedModule
from torch.distributed.fsdp import (
    FullOptimStateDictConfig,
    FullStateDictConfig,
    FullyShardedDataParallel,
    ShardedOptimStateDictConfig,
    ShardedStateDictConfig,
    StateDictType,
)
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.nn import Module
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Optimizer

from common.distributed import get_device, get_global_rank


def get_model_states(model: Module, *, sharded: bool = False):
    """
    Get model state dict.
    Call by all ranks.
    If full state dict, only use the result on rank 0.
    If sharded state dict, only for fsdp model.
    """
    if isinstance(model, OptimizedModule):
        model = model._orig_mod
    if isinstance(model, DistributedDataParallel):
        model = model.module
    if isinstance(model, FullyShardedDataParallel):
        configure_fsdp_states(model, sharded=sharded)
    return model.state_dict()


def get_optimizer_states(optimizer: Optimizer):
    """
    Get optimizer state dict.
    Call by all ranks. Only use the result on rank 0.
    """
    if isinstance(optimizer, ZeroRedundancyOptimizer):
        optimizer.consolidate_state_dict(to=0)
    return optimizer.state_dict() if get_global_rank() == 0 else None


def get_fsdp_optimizer_states(
    optimizer: Optimizer,
    model: FullyShardedDataParallel,
    *,
    sharded: bool = False,
):
    """
    Get fsdp optimizer state dict.
    Call by all ranks.
    If full state dict, only use the result on rank 0.
    If sharded state dict, only for fsdp model.
    """
    configure_fsdp_states(model, sharded=sharded)
    states = optimizer.state_dict()
    states = FullyShardedDataParallel.optim_state_dict(
        model=model,
        optim=optimizer,
        optim_state_dict=states,
    )
    return states


def set_fsdp_optimizer_states(
    states: Dict[str, torch.Tensor],
    optimizer: Optimizer,
    model: FullyShardedDataParallel,
):
    """
    Set fsdp optimizer state dict.
    Call by all ranks.
    """
    # Temporary Fix, convert ListConfig to List in states['param_groups'].
    # TODO(Jiashi): remove in the future.
    states["param_groups"] = [
        {k: list(v) if isinstance(v, ListConfig) else v for k, v in param_group.items()}
        for param_group in states["param_groups"]
    ]
    configure_fsdp_states(model, rank0_only=False)
    states = FullyShardedDataParallel.optim_state_dict_to_load(
        model=model,
        optim=optimizer,
        optim_state_dict=states,
    )
    optimizer.load_state_dict(states)


def configure_fsdp_states(
    model: FullyShardedDataParallel,
    *,
    rank0_only: bool = True,
    sharded: bool = False,
):
    """
    Configure fsdp state dict type.
    """
    if not sharded:
        FullyShardedDataParallel.set_state_dict_type(
            module=model,
            state_dict_type=StateDictType.FULL_STATE_DICT,
            state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=rank0_only),
            optim_state_dict_config=FullOptimStateDictConfig(
                offload_to_cpu=True, rank0_only=rank0_only
            ),
        )
    else:
        FullyShardedDataParallel.set_state_dict_type(
            module=model,
            state_dict_type=StateDictType.SHARDED_STATE_DICT,
            state_dict_config=ShardedStateDictConfig(offload_to_cpu=True),
            optim_state_dict_config=ShardedOptimStateDictConfig(offload_to_cpu=True),
        )


def fix_zero_optimizer_adamw_fused_states(optimizer: ZeroRedundancyOptimizer):
    """
    Fix loading issue for ZeroRedundancyOptimizer when used with AdamW.
    Issue: https://github.com/pytorch/pytorch/issues/124133
    For regular AdamW, ZeroRedundancyOptimizer keeps "step" on CPU, but this shouldn't
    be done for fused AdamW. So here we fix it.
    """
    for state in optimizer.optim.state.values():
        for name, value in state.items():
            state[name] = value.to(get_device())
