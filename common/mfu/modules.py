 # Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
 # SPDX-License-Identifier: Apache-2.0 

from abc import ABC, abstractmethod
from typing import Any, Callable, List
from torch import nn

from ..logger import get_logger
from .accumulator import FlopsAccumulator
from .basic_hooks import basic_flops_func
from .state import is_log_flop_enabled
from .utils import get_device_infos, get_model_run_state

logger = get_logger(__name__)


def get_flops_accumulator_hook(
    parent_module_name: str,
    flops_accumulator: FlopsAccumulator,
    flops_func: Callable,
):
    def _hook(module, args, kwargs, output):
        if is_log_flop_enabled():
            flops_accumulator(
                f"{parent_module_name}_{module.__class__.__name__}",
                (
                    flops_func(args, kwargs, output)
                    if isinstance(module, CustomFlops)
                    else flops_func(module, args, kwargs, output)
                ),
            )

    return _hook


class CustomFlops(ABC):
    """
    For functions,
    1. run the func within CustomFlops
    2. implement the hook `tflops`
    to support register_forward_hook
    """

    @abstractmethod
    def tflops(self, args, kwargs, output) -> float:
        pass


class Flops(nn.Module):
    """
    A wrapper for enable online mfu tracker.
    """

    def __init__(self, module: nn.Module):
        super().__init__()
        self._flops_wrap_module: nn.Module = module
        self.handlers = []
        self.tflops = 0
        self.flops_accumulator = FlopsAccumulator()
        self.mfu_factor = get_model_run_state(module).value
        logger.info(f"{module.__class__.__name__}: {get_model_run_state(module)}")

        self._register_flops_accumulator_hook()

    @property
    def module(self) -> nn.Module:
        """
        Returns the wrapped module (like :class:`DistributedDataParallel`).
        """
        return self._flops_wrap_module

    @staticmethod
    def unwrap_state_dict(state_dict):
        return {k.replace("_flops_wrap_module.", ""): v for k, v in state_dict.items()}

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.module(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.module, name)

    def __getitem__(self, key: int) -> Any:
        """Forward indexing calls in case the module is an ``nn.Sequential``."""
        return self.module.__getitem__(key)  # type: ignore[operator]

    def _register_flops_accumulator_hook(self):
        def _dfs_register_hooks(parent_name: str, cur_m: nn.Module):
            for m in cur_m.children():
                # custom hooks
                if isinstance(m, CustomFlops):
                    assert isinstance(m, nn.Module)
                    self.handlers.append(
                        m.register_forward_hook(
                            get_flops_accumulator_hook(
                                parent_name, self.flops_accumulator, m.tflops
                            ),
                            with_kwargs=True,
                        )
                    )
                    continue
                # built-in hooks
                is_registered = False
                for base_m, flops_func in basic_flops_func.items():
                    if (isinstance(base_m, str) and m.__class__.__name__ == base_m) or (
                        isinstance(base_m, type) and isinstance(m, base_m)
                    ):
                        self.handlers.append(
                            m.register_forward_hook(
                                get_flops_accumulator_hook(
                                    parent_name, self.flops_accumulator, flops_func
                                ),
                                with_kwargs=True,
                            )
                        )
                        is_registered = True
                        break
                if not is_registered:
                    if not isinstance(cur_m, (nn.ModuleList, nn.Sequential)):
                        parent_name = cur_m.__class__.__name__
                    _dfs_register_hooks(parent_name, m)

        _dfs_register_hooks("root", self.module)

    def summary(self, show: bool = False):
        if show:
            self.flops_accumulator.show()
        self.tflops = self.flops_accumulator.total()
        self.flops_accumulator.reset()

    def unwrap(self):
        for hdl in self.handlers:
            hdl.remove()


def get_mfu(iter_time, flop_modules: List[Flops], show: bool = False):
    # compute MFU
    _, ideal_TFLOPS = get_device_infos()
    achieve_TFLOPs = 0

    for module in flop_modules:
        assert isinstance(module, Flops)
        module.summary(show)
        achieve_TFLOPs += module.tflops * module.mfu_factor
        delattr(module, "tflops")

    mfu = achieve_TFLOPs / iter_time / ideal_TFLOPS
    # from https://arxiv.org/pdf/2001.08361
    # one PF-day = 10^15 × 24 × 3600 = 8.64 × 10^19 floating point operations.
    achieve_compute = achieve_TFLOPs / (10**3 * 24 * 3600)  # PF-days
    return {f"mfu({ideal_TFLOPS})": mfu}, {"compute": achieve_compute}, {"iter_time": iter_time}
