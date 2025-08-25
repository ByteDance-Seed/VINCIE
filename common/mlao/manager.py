 # Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
 # SPDX-License-Identifier: Apache-2.0 

import logging
import sys
import uuid
from itertools import chain, zip_longest
from typing import Dict, List, Optional
import torch
from tabulate import tabulate
from torch.utils._mode_utils import no_dispatch

from common.distributed import get_global_rank
from common.logger import get_logger

from .dataclass import FlatTensorMeta, MLAOTensorGroup
from .states import (
    get_bf16_pinned_buffer,
    get_fp32_pinned_buffer,
    get_mlao_root_id,
    get_mlao_stream,
    is_mlao_initialized,
)

_ACTIVATION_MANAGER = None


def set_activation_manager(**kwargs):
    global _ACTIVATION_MANAGER
    if is_mlao_initialized():
        if _ACTIVATION_MANAGER is not None:
            _ACTIVATION_MANAGER.reset()
        else:
            _ACTIVATION_MANAGER = ActivationManager(**kwargs)
    else:
        _ACTIVATION_MANAGER = None


def get_activation_manager():
    return _ACTIVATION_MANAGER


class ActivationManager:
    """
    Offload tensors saved by the forward pass to Pinned CPU Memory,
    then prefetch to GPU for backward, avoiding the costly recomputation.
    """

    def __init__(
        self,
        *,
        min_offload_mem: float = 1.0,
        max_alloc_mem_ratio: float = 0.6,
        prefetch_limit: int = 1,
        debug: bool = False,
    ):
        # Boundaries.
        self.min_offload_mem = min_offload_mem * 1024**2  # MiB
        self.max_alloc_mem = (
            torch.cuda.get_device_properties(0).total_memory * max_alloc_mem_ratio
        )  # B

        # Logger.
        self.logger = get_logger(self.__class__.__name__)
        if debug:
            self.logger.setLevel(logging.DEBUG)

        # Pinned CPU buffers.
        self.buffer: Dict[torch.dtype, torch.Tensor] = {
            torch.bfloat16: get_bf16_pinned_buffer(),
            torch.float32: get_fp32_pinned_buffer(),
        }
        self.buffer_offset: Dict[torch.dtype, int] = {torch.bfloat16: 0, torch.float32: 0}
        self.stream = get_mlao_stream()

        # Record tensor metas.
        self._tid_to_meta: Dict[str, FlatTensorMeta] = {}
        self._tid_to_device_tensor: Dict[str, torch.Tensor] = {}

        # For prefetching, we prefetch metas in reverse executive order.
        self._last_mlao_root_id = str(uuid.uuid4())
        self._exec_order_list: List[MLAOTensorGroup] = []
        self._mlao_root_id_to_exec_idx = {}
        self._prefetch_limit = max(0, prefetch_limit)

        self._is_first_iter = True
        self._is_first_prefetch = True
        self._prev_iter_last_mlao_root_id = str(uuid.uuid4())

        # For logging.
        self.packed_mem = 0
        self.skip_packed_mem = 0

    def get_statistics(self):
        mlao_statistics = {
            "mlao/cpu_mem_per_rank(GiB)": self.packed_mem / 1024**3,
            "mlao/gpu_mem_per_rank(GiB)": self.skip_packed_mem / 1024**3,
        }
        self.reset()
        return mlao_statistics

    def reset(self):
        self._prev_iter_last_mlao_root_id = self._last_mlao_root_id

        self.buffer_offset: Dict[torch.dtype, int] = {torch.bfloat16: 0, torch.float32: 0}

        self._tid_to_meta.clear()
        self._tid_to_device_tensor.clear()

        self._last_mlao_root_id = str(uuid.uuid4())
        self._exec_order_list.clear()
        self._mlao_root_id_to_exec_idx.clear()

        self._is_first_prefetch = True

        self.packed_mem = 0
        self.skip_packed_mem = 0

    def get_meta(self, tid: str) -> Optional[FlatTensorMeta]:
        return self._tid_to_meta.get(tid, None)

    def get_tensor(self, tid: str) -> Optional[torch.Tensor]:
        return self._tid_to_device_tensor.get(tid, None)

    def save_tensor(self, x: torch.Tensor):
        assert hasattr(x, "_tid")
        self._tid_to_device_tensor[x._tid] = x

    @staticmethod
    def get_tid(x: torch.Tensor) -> str:
        """
        Get the uuid of tensor, generate one and attach to it if not exist.
        """
        if not hasattr(x, "_tid"):
            x._tid = str(uuid.uuid4())
        return x._tid

    @staticmethod
    def mark_to_recompute(x: torch.Tensor) -> torch.Tensor:
        x._mlao_to_recompute = True
        return x

    @staticmethod
    def is_marked_to_recompute(x: torch.Tensor) -> bool:
        return getattr(x, "_mlao_to_recompute", False)

    def _record_exec_order(self, tid):
        cur_mlao_root_id = get_mlao_root_id()
        if self._last_mlao_root_id != cur_mlao_root_id:
            if self._exec_order_list:
                for record_tid in self._exec_order_list[-1].tids:
                    meta = self.get_meta(record_tid)
                    if meta.need_offload:
                        sys_ref_cnt = sys.getrefcount(meta.tensor)
                        self.logger.debug(
                            f"{record_tid} {meta.shape=} {sys_ref_cnt=} {meta.extra_ref_cnt=}"
                        )
                        corrected_ref_cnt = sys_ref_cnt - meta.extra_ref_cnt
                        if self._is_first_iter:
                            assert corrected_ref_cnt >= 2, f"Detected useless save. {meta.shape=}"
                        if corrected_ref_cnt == 2:
                            meta.deallocate()
                if get_global_rank() == 0 and self._is_first_iter:
                    self.viz_activations(self._exec_order_list[-1].tids)
            self._exec_order_list.append(MLAOTensorGroup(cur_mlao_root_id))
            self._last_mlao_root_id = cur_mlao_root_id
        self._exec_order_list[-1].tids.append(tid)
        self._mlao_root_id_to_exec_idx[cur_mlao_root_id] = len(self._exec_order_list) - 1

    def maybe_pack_to_cpu(
        self, tensor: torch.Tensor, *, pack_on_device: bool = False, extra_ref_cnt: int = 0
    ):
        tid = self.get_tid(tensor)

        # If already packed, increase reference count and return.
        if (meta := self.get_meta(tid)) is not None:
            meta.ref_cnt += 1
            meta.extra_ref_cnt += extra_ref_cnt
            self._record_exec_order(tid)
            return tid

        # Get where to put the data in pinned_memory.
        need_offload = not pack_on_device and self._maybe_offload(tensor)
        if need_offload:
            offset_st = self.buffer_offset[tensor.dtype]
            offset_ed = offset_st + tensor.numel()
            self.buffer_offset[tensor.dtype] += tensor.numel()
        else:
            offset_st = offset_ed = 0

        # Get metadata for rebuilding the tensor.
        meta = FlatTensorMeta(
            tensor=tensor,
            offsets=(offset_st, offset_ed),
            need_offload=need_offload,
            to_recompute=self.is_marked_to_recompute(tensor),
            extra_ref_cnt=extra_ref_cnt,
        )
        self._tid_to_meta[tid] = meta

        # Offload to cpu.
        tmem = tensor.numel() * tensor.element_size()
        if meta.need_offload:
            cpu_tensor = self.buffer[meta.dtype][meta.offsets[0] : meta.offsets[1]]

            self.stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self.stream):
                cpu_tensor.copy_(tensor.detach().flatten(), non_blocking=True)

                # In case of offloading, `tensor.data` is created on the main stream.
                # We need to hand it over to the activation stream for non_blocking offload.
                with no_dispatch():
                    tensor.data.record_stream(self.stream)
                meta.offload_event = self.stream.record_event()
            self.packed_mem += tmem
        elif not meta.to_recompute:
            self._tid_to_device_tensor[tid] = tensor
            if tensor.is_cuda:
                self.skip_packed_mem += tmem

        self._record_exec_order(tid)
        return tid

    def maybe_unpack_from_cpu(self, tid):
        # Get tensor's meta
        meta = self.get_meta(tid)
        assert meta is not None

        if self.get_tensor(tid) is None:
            self.logger.info(
                f"[UnPack] {tid=} {meta.need_offload=} {meta.to_recompute=} NOT Prefetched :("
            )
            # Make sure offload is complete.
            meta.finish_offload()
            tensor = self.buffer[meta.dtype][meta.offsets[0] : meta.offsets[1]]
            self.stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self.stream):
                self._tid_to_device_tensor[tid] = tensor.to(device=meta.device, non_blocking=True)
                meta.upload_event = self.stream.record_event()

        # Fetch to device.
        tensor = self.get_tensor(tid)

        # Make sure upload is complete.
        meta.finish_upload()
        tensor = tensor.view(meta.shape)

        # Reallocate saved tensor.
        meta.reallocate(tensor)
        tensor = meta.tensor

        # Update reference counter.
        meta.ref_cnt -= 1
        if meta.ref_cnt == 0:
            self._tid_to_device_tensor.pop(tid)
            meta.tensor = None

        return tensor

    def _maybe_offload(self, tensor: torch.Tensor) -> bool:
        if get_mlao_root_id() == self._prev_iter_last_mlao_root_id:
            return False

        if self.is_marked_to_recompute(tensor):
            return False

        if tensor.is_cpu:
            return False

        if tensor.dtype not in [torch.float32, torch.bfloat16]:
            return False

        occupy_mem = tensor.numel() * tensor.element_size()
        if occupy_mem < self.min_offload_mem:
            return False

        if self.buffer_offset[tensor.dtype] + tensor.numel() > self.buffer[tensor.dtype].numel():
            self.logger.warning(f"[Skip] Not enough buffer to pack tensor {tensor.dtype=}.")
            return False

        return True

    def prefetch(self):
        cur_idx = self._mlao_root_id_to_exec_idx[get_mlao_root_id()]

        if self._is_first_prefetch and self._exec_order_list:
            # We don't deallocate tensors for the last checkpointing scope
            # as it will be used instantly.
            for record_tid in self._exec_order_list[-1].tids:
                meta = self.get_meta(record_tid)
                if meta.need_offload:
                    meta.need_offload = False
                    # Although the offload is issued, we don't care whether it finish or not.
                    meta.offload_event = None
                    self._tid_to_device_tensor[record_tid] = meta.tensor
                    if meta.tensor.is_cuda:
                        self.skip_packed_mem += meta.numel * meta.dtype.itemsize
            self._is_first_prefetch = False

        if self._is_first_iter:
            if get_global_rank() == 0 and self._exec_order_list:
                self.viz_activations(self._exec_order_list[-1].tids)
            self._is_first_iter = False

        alloc_mem = None
        next_tensor_mem = 0
        target_index = cur_idx - 1
        for _ in range(self._prefetch_limit):
            if target_index < 0:
                return

            if self._exec_order_list[target_index].tids:
                if alloc_mem is None:
                    # Delay this inefficient call until it necessary.
                    alloc_mem = torch.cuda.memory_allocated()

                if (alloc_mem + next_tensor_mem) > self.max_alloc_mem:
                    return

                alloc_mem, next_tensor_mem = self._prefetch_tensor_group(
                    self._exec_order_list[target_index], alloc_mem
                )

            target_index -= 1

    def _prefetch_tensor_group(self, tensor_group, cur_alloc_mem):
        while tensor_group.tids:
            tid = tensor_group.tids[0]

            # [SKIP] Duplicated fetch.
            if self.get_tensor(tid) is not None:
                tensor_group.tids.pop(0)
                continue

            meta = self.get_meta(tid)

            # [SKIP] Will recompute, no need to fetch.
            if meta.to_recompute:
                tensor_group.tids.pop(0)
                continue

            # [SKIP] Memory is on the edge.
            tensor_mem = meta.numel * meta.dtype.itemsize
            if cur_alloc_mem + tensor_mem > self.max_alloc_mem:
                self.logger.debug(f"[Prefetch {tid=}] Hit cuda memory limit, skip prefetch.")
                return cur_alloc_mem, tensor_mem

            cur_alloc_mem += tensor_mem

            # [Prefetch]
            # Make sure offload is complete.
            meta.finish_offload()
            tensor = self.buffer[meta.dtype][meta.offsets[0] : meta.offsets[1]]
            tensor_group.tids.pop(0)
            self.stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self.stream):
                self._tid_to_device_tensor[tid] = tensor.to(device=meta.device, non_blocking=True)
                meta.upload_event = self.stream.record_event()

        return cur_alloc_mem, 0

    def viz_activations(self, tids):
        N_COLS = 7
        datas = []
        visited = []
        for tid in tids:
            if tid in visited:
                continue
            visited.append(tid)
            meta = self.get_meta(tid)
            tmem = meta.numel * meta.dtype.itemsize / 1024**2
            datas.extend(
                [
                    tid,
                    meta.shape,
                    meta.dtype,
                    tmem,
                    meta.ref_cnt,
                    meta.on_device,
                ]
            )
            datas.append(0.0 if meta.to_recompute else tmem)
        data = list(chain(*[datas]))
        total_activations_mem = sum(data[3::N_COLS])
        total_mem_to_save = sum(data[6::N_COLS])
        data.extend([None] * (N_COLS - (len(data) % N_COLS)))
        data.extend(
            [
                f"rank{get_global_rank()} total",
                "",
                "",
                total_activations_mem,
                "",
                "",
                total_mem_to_save,
            ]
        )
        data = zip_longest(*[data[i::N_COLS] for i in range(N_COLS)])
        table = tabulate(
            data,
            headers=[
                "tid",
                "shape",
                "dtype",
                "activations(MiB)",
                "ref_cnt",
                "on device",
                "saved activations(MiB)",
            ],
            tablefmt="pipe",
            numalign="left",
            stralign="center",
        )
        self.logger.info(f"\n{table}")
