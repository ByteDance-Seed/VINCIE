 # Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
 # SPDX-License-Identifier: Apache-2.0 

from typing import Optional
import torch

from common.logger import get_logger

_BF16_PINNED_BUFFER = None
_FP32_PINNED_BUFFER = None

_MLAO_STREAM = None

_MLAO_ROOT_ID = None
_RECOMPUTING = None


def is_mlao_initialized() -> bool:
    return _BF16_PINNED_BUFFER is not None


def get_bf16_pinned_buffer() -> torch.Tensor:
    assert _BF16_PINNED_BUFFER is not None
    return _BF16_PINNED_BUFFER


def get_fp32_pinned_buffer() -> torch.Tensor:
    assert _FP32_PINNED_BUFFER is not None
    return _FP32_PINNED_BUFFER


def get_mlao_stream() -> Optional[torch.cuda.Stream]:
    return _MLAO_STREAM


def set_mlao_root_id(root_id: Optional[str]):
    global _MLAO_ROOT_ID
    _MLAO_ROOT_ID = root_id


def get_mlao_root_id() -> Optional[str]:
    return _MLAO_ROOT_ID


def set_recomputing(value: Optional[bool]):
    global _RECOMPUTING
    _RECOMPUTING = value


def get_recomputing() -> Optional[bool]:
    return _RECOMPUTING


def is_recomputing() -> bool:
    return _RECOMPUTING is True


def in_checkpointing() -> bool:
    return _RECOMPUTING is not None


def initialize_states(fp32_size: int, bf16_size: int):
    global _BF16_PINNED_BUFFER
    global _FP32_PINNED_BUFFER
    global _MLAO_STREAM

    logger = get_logger(__name__)
    logger.info("Allocating BF16 buffer.")
    _BF16_PINNED_BUFFER = torch.empty(bf16_size * 1024**3, dtype=torch.bfloat16, pin_memory=True)
    logger.info("Done.")

    logger.info("Allocating FP32 buffer.")
    _FP32_PINNED_BUFFER = torch.empty(fp32_size * 1024**3, dtype=torch.float32, pin_memory=True)
    logger.info("Done.")

    _MLAO_STREAM = torch.cuda.Stream()
