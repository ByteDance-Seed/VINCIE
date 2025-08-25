 # Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
 # SPDX-License-Identifier: Apache-2.0 

"""
Set/Get states.
"""

from torch.utils._contextlib import _NoParamDecoratorContextManager

_ENABLE_LOG_FLOP = False


def set_enable_log_flop(enable: bool):
    global _ENABLE_LOG_FLOP
    _ENABLE_LOG_FLOP = enable


def is_log_flop_enabled():
    return _ENABLE_LOG_FLOP


class enable_flops_accumulate(_NoParamDecoratorContextManager):
    def __enter__(self) -> None:
        set_enable_log_flop(True)

    def __exit__(self, exc_type=None, exc_value=None, traceback=None) -> None:
        set_enable_log_flop(False)
