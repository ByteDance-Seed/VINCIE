 # Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
 # SPDX-License-Identifier: Apache-2.0 

import torch

from .manager import get_activation_manager
from .states import get_recomputing, set_recomputing


class MLAOSavedTensorsHooks(torch.autograd.graph.saved_tensors_hooks):
    def __init__(self, pack_on_device=False):
        def pack_hook(x):
            # This is a hack, do not pack parameters.
            if x._base is not None:
                return x
            return get_activation_manager().maybe_pack_to_cpu(x, pack_on_device=pack_on_device)

        def unpack_hook(tid):
            if torch.is_tensor(tid):
                return tid
            return get_activation_manager().maybe_unpack_from_cpu(tid)

        super().__init__(pack_hook, unpack_hook)


class RecomputingCtx:
    def __init__(self, recomputing: bool):
        self.recomputing = recomputing

    def __enter__(self) -> None:
        self.prev = get_recomputing()
        set_recomputing(self.recomputing)

    def __exit__(self, exc_type=None, exc_value=None, traceback=None) -> None:
        set_recomputing(self.prev)
