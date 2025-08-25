 # Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
 # SPDX-License-Identifier: Apache-2.0 

from .manager import set_activation_manager
from .states import initialize_states


def initialize_mlao(
    *,
    fp32_size: int,
    bf16_size: int,
    min_offload_mem: float = 1.0,
    max_alloc_mem_ratio: float = 0.6,
    prefetch_limit: int = 1,
    debug: bool = False,
):
    """
    :param fp32_size: Pin (fp32_size * 4)GiB CPU memory for offloading fp32 tensors.
    :param bf16_size: Pin (bf16_size * 2)GiB CPU memory for offloading bf16 tensors.
    :param min_offload_mem: Threshold(MiB) to decide whether offload the activation to cpu.
           (default = 1.0 MiB)
    :param max_alloc_mem_ratio: Threshold(MiB) to decide whether stop current
           activation prefetching to prevent OOM. (default = 0.6, means
           stop prefetching when over 60% of cuda memory is allocated.)
    :param prefetch_limit: max number of checkpointing regions to prefetch. (default = 1)
    :param debug: whether to print debug messages. (default = False)
    """
    initialize_states(fp32_size, bf16_size)
    set_activation_manager(
        min_offload_mem=min_offload_mem,
        max_alloc_mem_ratio=max_alloc_mem_ratio,
        prefetch_limit=prefetch_limit,
        debug=debug,
    )
