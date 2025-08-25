 # Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
 # SPDX-License-Identifier: Apache-2.0 

"""
Advanced distributed functions for sequence parallel.
"""

from typing import Optional
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import ShardingStrategy

from .basic import get_global_rank, get_world_size

_DATA_PARALLEL_GROUP = None
_SEQUENCE_PARALLEL_GROUP = None
_SEQUENCE_PARALLEL_CPU_GROUP = None
_MODEL_SHARD_CPU_INTER_GROUP = None
_MODEL_SHARD_CPU_INTRA_GROUP = None
_MODEL_SHARD_INTER_GROUP = None
_MODEL_SHARD_INTRA_GROUP = None
_LOAD_BALANCE_CPU_GROUP = None
_LOAD_BALANCE_GPU_GROUP = None


def get_data_parallel_group() -> Optional[dist.ProcessGroup]:
    """
    Get data parallel process group.
    """
    return _DATA_PARALLEL_GROUP


def get_sequence_parallel_group() -> Optional[dist.ProcessGroup]:
    """
    Get sequence parallel process group.
    """
    return _SEQUENCE_PARALLEL_GROUP


def get_sequence_parallel_cpu_group() -> Optional[dist.ProcessGroup]:
    """
    Get sequence parallel CPU process group.
    """
    return _SEQUENCE_PARALLEL_CPU_GROUP


def get_data_parallel_rank() -> int:
    """
    Get data parallel rank.
    """
    group = get_data_parallel_group()
    return dist.get_rank(group) if group else get_global_rank()


def get_data_parallel_world_size() -> int:
    """
    Get data parallel world size.
    """
    group = get_data_parallel_group()
    return dist.get_world_size(group) if group else get_world_size()


def get_sequence_parallel_rank() -> int:
    """
    Get sequence parallel rank.
    """
    group = get_sequence_parallel_group()
    return dist.get_rank(group) if group else 0


def get_sequence_parallel_world_size() -> int:
    """
    Get sequence parallel world size.
    """
    group = get_sequence_parallel_group()
    return dist.get_world_size(group) if group else 1


def get_model_shard_cpu_intra_group() -> Optional[dist.ProcessGroup]:
    """
    Get the CPU intra process group of model sharding.
    """
    return _MODEL_SHARD_CPU_INTRA_GROUP


def get_model_shard_cpu_inter_group() -> Optional[dist.ProcessGroup]:
    """
    Get the CPU inter process group of model sharding.
    """
    return _MODEL_SHARD_CPU_INTER_GROUP


def get_model_shard_intra_group() -> Optional[dist.ProcessGroup]:
    """
    Get the GPU intra process group of model sharding.
    """
    return _MODEL_SHARD_INTRA_GROUP


def get_model_shard_inter_group() -> Optional[dist.ProcessGroup]:
    """
    Get the GPU inter process group of model sharding.
    """
    return _MODEL_SHARD_INTER_GROUP


def get_load_balance_cpu_group() -> Optional[dist.ProcessGroup]:
    """
    Get the CPU process group for load_balance.
    """
    return _LOAD_BALANCE_CPU_GROUP


def get_load_balance_gpu_group() -> Optional[dist.ProcessGroup]:
    """
    Get the GPU process group for load_balance.
    """
    return _LOAD_BALANCE_GPU_GROUP


def init_sequence_parallel(sequence_parallel_size: int):
    """
    Initialize sequence parallel.
    """
    global _DATA_PARALLEL_GROUP
    global _SEQUENCE_PARALLEL_GROUP
    global _SEQUENCE_PARALLEL_CPU_GROUP
    assert dist.is_initialized()
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    data_parallel_size = world_size // sequence_parallel_size
    for i in range(data_parallel_size):
        start_rank = i * sequence_parallel_size
        end_rank = (i + 1) * sequence_parallel_size
        ranks = range(start_rank, end_rank)
        group = dist.new_group(ranks)
        cpu_group = dist.new_group(ranks, backend="gloo")
        if rank in ranks:
            _SEQUENCE_PARALLEL_GROUP = group
            _SEQUENCE_PARALLEL_CPU_GROUP = cpu_group


def init_model_shard_group(
    *,
    sharding_strategy: ShardingStrategy,
    device_mesh: Optional[DeviceMesh] = None,
):
    """
    Initialize process group of model sharding.
    """
    global _MODEL_SHARD_INTER_GROUP
    global _MODEL_SHARD_INTRA_GROUP
    global _MODEL_SHARD_CPU_INTER_GROUP
    global _MODEL_SHARD_CPU_INTRA_GROUP
    assert dist.is_initialized()
    world_size = dist.get_world_size()
    if device_mesh is not None:
        num_shards_per_group = device_mesh.shape[1]
    elif sharding_strategy == ShardingStrategy.NO_SHARD:
        num_shards_per_group = 1
    elif sharding_strategy in [
        ShardingStrategy.HYBRID_SHARD,
        ShardingStrategy._HYBRID_SHARD_ZERO2,
    ]:
        num_shards_per_group = torch.cuda.device_count()
    else:
        num_shards_per_group = world_size
    num_groups = world_size // num_shards_per_group
    device_mesh = torch.arange(get_world_size()).view(num_groups, num_shards_per_group)

    for ranks in device_mesh.tolist():
        gpu_group = dist.new_group(ranks, backend="nccl")
        cpu_group = dist.new_group(ranks, backend="gloo")
        if get_global_rank() in ranks:
            _MODEL_SHARD_INTRA_GROUP = gpu_group
            _MODEL_SHARD_CPU_INTRA_GROUP = cpu_group

    for ranks in device_mesh.t().tolist():
        gpu_group = dist.new_group(ranks, backend="nccl")
        cpu_group = dist.new_group(ranks, backend="gloo")
        if get_global_rank() in ranks:
            _MODEL_SHARD_INTER_GROUP = gpu_group
            _MODEL_SHARD_CPU_INTER_GROUP = cpu_group


def init_load_balance_process_group(
    balance_group_size: int = 8,
):
    """
    Initialize process group for load_balance.
    """
    global _LOAD_BALANCE_CPU_GROUP
    global _LOAD_BALANCE_GPU_GROUP
    assert dist.is_initialized()
    _LOAD_BALANCE_GPU_GROUP, _ = dist.new_subgroups(balance_group_size, backend="nccl")
    _LOAD_BALANCE_CPU_GROUP, _ = dist.new_subgroups(balance_group_size, backend="gloo")
