 # Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
 # SPDX-License-Identifier: Apache-2.0 

import collections
import uuid
from dataclasses import dataclass
from typing import Any
import torch
from torch import nn
from torch.distributed.utils import _apply_to_tensors
from torch.utils._pytree import tree_flatten

from ..logger import get_logger
from .contexts import MLAOSavedTensorsHooks
from .dataclass import MLAOStateConfig
from .manager import get_activation_manager
from .states import get_mlao_root_id, in_checkpointing, is_recomputing, set_mlao_root_id

__all__ = ["MLAOState"]


logger = get_logger(__name__)


@dataclass
class MLAOSharedContext:
    root_id: str
    prefetched: bool = False


class MLAOState:
    def __init__(self, module: nn.Module, config: MLAOStateConfig):
        self._is_root = None
        self._module = module
        self._config = config
        self._shared_ctx = None

        self._saved_inputs_deque = collections.deque()
        self._saved_outputs_deque = collections.deque()
        self._saved_input_tids_deque = collections.deque()
        self._module_outputs_deque = collections.deque()
        self._rng_state_recover_fn_deque = collections.deque()

        # Prepend to make sure capture checkpoint inputs.
        self._pre_forward_handle = module.register_forward_pre_hook(
            self._pre_forward, prepend=self._config.pack_inputs, with_kwargs=True
        )
        self._post_forward_handle = module.register_forward_hook(self._post_forward)

    def _lazy_init(self):
        if self._is_root is not None:
            return  # no-op: already initialized

        self._is_root = True
        assert self._shared_ctx is None
        self._shared_ctx = MLAOSharedContext(root_id=str(uuid.uuid4()))
        for sub_module in self._module.modules():
            if sub_module is self._module:
                continue
            if (state := getattr(sub_module, "mlao_state", None)) is not None:
                assert state._is_root is None
                state._is_root = False
                state._shared_ctx = self._shared_ctx

    def _valid_to_save(self):
        return (
            get_activation_manager() is not None and torch.is_grad_enabled() and in_checkpointing()
        )

    def recover_outputs_and_rng_states(self):
        outputs = self._module_outputs_deque.popleft()
        # Recover torch rng states.
        self._rng_state_recover_fn_deque.popleft()()
        return outputs

    def save_outputs_and_rng_states(self, outputs):
        self._module_outputs_deque.append(outputs)

        # Preserve torch rng states after forward.
        cpu_state = torch.get_rng_state()
        gpu_state = torch.cuda.get_rng_state()

        def rng_state_recover_fn():
            torch.set_rng_state(cpu_state)
            torch.cuda.set_rng_state(gpu_state)

        self._rng_state_recover_fn_deque.append(rng_state_recover_fn)

    def _pre_forward(self, module, args, kwargs):
        if not self._valid_to_save():
            return args, kwargs

        self._lazy_init()

        self.prev_root_id = get_mlao_root_id()
        set_mlao_root_id(self._shared_ctx.root_id)

        MLAOSavedTensorsHooks(self._config.pack_on_device).__enter__()

        if is_recomputing():
            args, kwargs = self.unpack_inputs(args, kwargs)
        elif in_checkpointing():
            args, kwargs = self.pack_inputs(args, kwargs)

        if not self._config.pack_activations:
            MLAOSavedTensorsHooks().__exit__(None, None, None)

        return args, kwargs

    def _post_forward(self, module, inputs, outputs):
        if not self._valid_to_save():
            return outputs

        if is_recomputing():
            outputs = self.unpack_outputs(outputs)
        elif in_checkpointing():
            outputs = self.pack_outputs(outputs)
            if self._is_root:
                outputs = self._register_pre_backward_hook(outputs)
            if self._config.pack_activations:
                self.save_outputs_and_rng_states(outputs)

        if self._config.pack_activations:
            MLAOSavedTensorsHooks().__exit__(None, None, None)

        set_mlao_root_id(self.prev_root_id)

        return outputs

    def _pre_backward(self, grad: torch.Tensor) -> torch.Tensor:
        # Register prefetch hook on all output tensors,
        # while only let the first issued tensor hook to actually perform prefetch.
        if self._shared_ctx.prefetched:
            return grad

        self.prev_root_id = get_mlao_root_id()
        set_mlao_root_id(self._shared_ctx.root_id)
        # Time to prefetch.
        if (manager := get_activation_manager()) is not None:
            manager.prefetch()
            self._shared_ctx.prefetched = True
        set_mlao_root_id(self.prev_root_id)
        return grad

    def _register_pre_backward_hook(self, output: Any) -> Any:
        if not torch.is_grad_enabled():
            return output
        flat_outputs, _ = tree_flatten(output)
        hooked = False
        for t in flat_outputs:
            if torch.is_tensor(t) and t.requires_grad:
                t.register_hook(self._pre_backward)
                hooked = True
        assert hooked, "Fail to register prefetch hook on output."
        self._shared_ctx.prefetched = False
        return output

    def pack_inputs(self, args, kwargs):
        if not self._config.pack_inputs:
            # [Note #1] To set the recompute tensor with packed tid for unpack,
            # In pack_inputs: Record tensor's tid in the first forward pass
            # In unpack_inputs:
            #       1. Restore tid to tensor in the recompute pass
            #       2. Save the recomputed tensor in manager
            # [Note #2] pack_inputs=True indicates that we are processing
            # checkpointing inputs and is unexpected to recompute.
            self._saved_input_tids_deque.append([])

            def save_tensor_tid(x):
                if self._config.pack_activations and x.requires_grad:
                    x = x.contiguous()
                    # Not support packing view tensor, call x.clone() before
                    # enter MLAOSavedTensorsHooks.pack_hook.
                    if x._base is not None:
                        x = x.clone()
                    x = get_activation_manager().mark_to_recompute(x)
                    self._saved_input_tids_deque[-1].append(get_activation_manager().get_tid(x))
                else:
                    self._saved_input_tids_deque[-1].append(None)
                return x

            args = _apply_to_tensors(save_tensor_tid, args)
            kwargs = _apply_to_tensors(save_tensor_tid, kwargs)

            return args, kwargs

        self._saved_inputs_deque.append([])

        def pack_tensor(x):
            if x.requires_grad:
                x = x.contiguous()
                self._saved_inputs_deque[-1].append(
                    get_activation_manager().maybe_pack_to_cpu(
                        x, pack_on_device=self._config.pack_on_device, extra_ref_cnt=1
                    )
                )
            else:
                self._saved_inputs_deque[-1].append(None)
            return x

        args = _apply_to_tensors(pack_tensor, args)
        kwargs = _apply_to_tensors(pack_tensor, kwargs)

        return args, kwargs

    def unpack_inputs(self, args, kwargs):
        if not self._config.pack_inputs:
            # See [Note #1]
            _saved_input_tids = self._saved_input_tids_deque.popleft()

            def restore_tensor_tid(x):
                if _saved_input_tids and (tid := _saved_input_tids.pop(0)) is not None:
                    if (meta := get_activation_manager().get_meta(tid)) is not None:
                        x._tid = tid
                        meta.check_tensor_match(x)
                        get_activation_manager().save_tensor(x)
                    else:
                        logger.warning("Marked input is not the packed input.")
                return x

            args = _apply_to_tensors(restore_tensor_tid, args)
            kwargs = _apply_to_tensors(restore_tensor_tid, kwargs)
            assert len(_saved_input_tids) == 0
            return args, kwargs

        _saved_inputs = self._saved_inputs_deque.pop()

        def unpack_tensor(x):
            saved = _saved_inputs.pop(0)
            if saved is None:
                return x
            return get_activation_manager().maybe_unpack_from_cpu(saved)

        args = _apply_to_tensors(unpack_tensor, args)
        kwargs = _apply_to_tensors(unpack_tensor, kwargs)

        assert len(_saved_inputs) == 0
        return args, kwargs

    def pack_outputs(self, outputs):
        if not self._config.pack_activations:
            return outputs

        self._saved_outputs_deque.append([])

        def pack_tensor(x):
            if x.requires_grad:
                x = x.contiguous()
                self._saved_outputs_deque[-1].append(
                    get_activation_manager().maybe_pack_to_cpu(
                        x, pack_on_device=self._config.pack_on_device, extra_ref_cnt=1
                    )
                )
            else:
                self._saved_outputs_deque[-1].append(None)
            return x

        return _apply_to_tensors(pack_tensor, outputs)

    def unpack_outputs(self, outputs):
        if not self._config.pack_activations:
            return outputs
        _saved_outputs = self._saved_outputs_deque.popleft()

        def unpack_tensor(x):
            saved = _saved_outputs.pop(0)
            if saved is None:
                return x
            return get_activation_manager().maybe_unpack_from_cpu(saved)

        outputs = _apply_to_tensors(unpack_tensor, outputs)
        assert len(_saved_outputs) == 0
        return outputs
