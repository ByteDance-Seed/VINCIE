 # Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
 # SPDX-License-Identifier: Apache-2.0 

"""
Flops accumulator for MFU logging.
"""

import logging
from itertools import chain, zip_longest
from typing import Dict, Union
import torch
from tabulate import tabulate

from common.logger import get_logger

logger = get_logger(__name__)


class FlopsAccumulator:
    """
    Accumulate total FLOPs each iteration.
    This is intended for mfu logging purposes.
    """

    def __init__(self, disable=False):
        self.flops_accumulator: Dict[str, float] = {}
        self.disable = disable

    def reset(self):
        """
        Reset the accumulator.
        """
        self.flops_accumulator = {}

    def __call__(self, key: str, flops: Union[float, torch.Tensor]):
        if self.disable:
            return
        if key not in self.flops_accumulator:
            self.flops_accumulator[key] = 0.0
        if logger.level <= logging.DEBUG:
            # NB: This assertion will call a cudaStreamSync.
            assert flops >= 0, f"{flops} out of bound"
        self.flops_accumulator[key] += flops

    def total(self):
        res = sum(self.flops_accumulator.values())
        if torch.is_tensor(res):
            res = res.item()
        return res

    def show(self):
        N_COLS = 2
        data = list(chain(*[(k, v) for k, v in self.flops_accumulator.items()]))
        tflops = sum(data[1::2])
        data.extend([None] * (N_COLS - (len(data) % N_COLS)))
        data.extend(["total", tflops])
        data = zip_longest(*[data[i::N_COLS] for i in range(N_COLS)])
        table = tabulate(
            data,
            headers=["Name", "TFLOPs"] * (N_COLS // 2),
            tablefmt="pipe",
            numalign="left",
            stralign="center",
        )
        logger.info(f"\n{table}")
