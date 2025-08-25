 # Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
 # SPDX-License-Identifier: Apache-2.0 

"""
Utility functions.
"""

from enum import Enum
import torch

from ..logger import get_logger
from ..precision import bf16_enabled, tf32_enabled


class RunState(Enum):
    BF16_EVAL = 1
    TF32_EVAL = 2
    FP32_EVAL = 16
    BF16_TRAIN = 3
    TF32_TRAIN = 6
    FP32_TRAIN = 48


def get_model_run_state(model: torch.nn.Module) -> RunState:
    is_train = model.training
    if bf16_enabled():
        state = RunState.BF16_TRAIN if is_train else RunState.BF16_EVAL
    elif tf32_enabled():
        state = RunState.TF32_TRAIN if is_train else RunState.TF32_EVAL
    else:
        state = RunState.FP32_TRAIN if is_train else RunState.FP32_EVAL
    return state


def get_device_infos():
    peak_tflops = -1
    arch = torch.cuda.get_device_capability()
    if arch[0] == 8 and arch[1] == 0:  # A100/A800
        peak_tflops = 312
        gpu_type = "A800"
    elif arch[0] == 9 and arch[1] == 0:  # H100/H800
        peak_tflops = 989
        gpu_type = "H800"
    else:
        logger = get_logger(__name__)
        logger.warning(f"unknown default tflops of device capability {arch[0]}.{arch[1]}")
    return gpu_type, peak_tflops
