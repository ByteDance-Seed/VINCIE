 # Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
 # SPDX-License-Identifier: Apache-2.0 

import types
import torch.nn as nn

from .dataclass import MLAOStateConfig
from .mlao_state import MLAOState
from .states import is_recomputing

__all__ = ["mlao_skip_recompute", "mlao_wrap"]


def _mlao_wrap(
    module: nn.Module,
    *,
    config: MLAOStateConfig,
):
    """
    Attach a MLAOState to the module.
    """
    assert not hasattr(module, "mlao_state")
    module.mlao_state = MLAOState(module, config)

    if config.pack_activations:

        def mlao_forward(self, *args, **kwargs):
            if is_recomputing():
                outputs = module.mlao_state.recover_outputs_and_rng_states()
            else:
                # Run original forward.
                outputs = module.mlao_inner_forward(*args, **kwargs)
            return outputs

        module.mlao_inner_forward = module.forward
        # Official way to attaching a different `forward` method.
        # https://github.com/pytorch/pytorch/blob/38d9924bfc07fcdcf2893492a1235be8e9ffee70/test/test_jit.py#L9838
        module.forward = types.MethodType(mlao_forward, module)

    return module


def mlao_wrap(module: nn.Module, pack_on_device: bool = False):
    """
    Offloads input tensors of torch.utils.checkpointing wrapped module
    for saving GPU memory.
    """
    return _mlao_wrap(
        module, config=MLAOStateConfig(pack_inputs=True, pack_on_device=pack_on_device)
    )


def mlao_skip_recompute(module: nn.Module, pack_on_device: bool = False):
    """
    Enable async activation offloading & retrieve for the wrapped nodule
    for skipping recompute in activation checkpointing.
    """
    return _mlao_wrap(
        module, config=MLAOStateConfig(pack_activations=True, pack_on_device=pack_on_device)
    )
