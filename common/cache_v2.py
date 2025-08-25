 # Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
 # SPDX-License-Identifier: Apache-2.0 

from typing import Callable, Hashable, Sequence
import torch

from common.logger import get_logger

logger = get_logger(__name__)


def use_if_not_none(x, default):
    return x if x is not None else default


class Cache:
    """Caching reusable args for faster training"""

    def __init__(
        self,
        disable=False,
        enable_precompute=False,
        prefix="",
        cache=None,
        src_to_idx=None,
        record_src=None,
        record_functions=None,
    ):
        self.disable = disable
        self.enable_precompute = enable_precompute
        self.prefix = prefix
        self.cache = use_if_not_none(cache, {})
        self.src_to_idx = use_if_not_none(src_to_idx, {})
        self.record_src = use_if_not_none(record_src, [])
        self.record_functions = use_if_not_none(record_functions, [])
        self.prev_num_records = None  # Only use by the outermost namespace.

    def reset(self, disable_cache: bool = False):
        self.disable = disable_cache
        self.cache.clear()
        self.src_to_idx.clear()
        self.record_src.clear()
        if self.enable_precompute and self.prev_num_records == 0:
            logger.info(f"Record cache keys: {[r[0] for r in self.record_functions]}")
        self.prev_num_records = len(self.record_functions)

    def mark_src(self, src: torch.Tensor):
        if self.disable or (not self.enable_precompute):
            return
        assert torch.is_tensor(src)
        self.src_to_idx[src] = len(self.record_src)
        self.record_src.append(src)

    def precompute(self):
        if self.disable or (not self.enable_precompute):
            return
        try:
            for key, fn, tids in self.record_functions:
                result = fn(*[self.record_src[tid] for tid in tids])
                self._extend_cache_src(result)
                self.cache[key] = result
        except Exception as e:
            logger.warning(f"Fail on recompute {key=} {fn=} {tids=} {e}")
            self.record_functions.clear()
            self.reset()

    def __call__(self, key: str, fn: Callable, *args):
        if self.disable:
            return fn(*args)

        key = self.prefix + key
        try:
            result = self.cache[key]
        except KeyError:
            result = fn(*args)
            if self.enable_precompute and (tids := self._get_src_ids(args)) is not None:
                logger.debug(f"Recording {key=} {fn=} {tids=}")
                self.record_functions.append([key, fn, tids])
                self._extend_cache_src(result)
            self.cache[key] = result
        return result

    def _extend_cache_src(self, result):
        if not isinstance(result, Sequence):
            result = (result,)
        for res in result:
            if torch.is_tensor(res):
                self.mark_src(res)

    def _get_src_ids(self, args):
        # For lambda functions without args, fallback to cache_v1.
        if len(args):
            src_ids = []
            for arg in args:
                if isinstance(arg, Hashable) and arg in self.src_to_idx:
                    src_ids.append(self.src_to_idx[arg])
            if len(src_ids) == len(args):
                return src_ids
        return None

    def namespace(self, namespace: str):
        return Cache(
            disable=self.disable,
            enable_precompute=self.enable_precompute,
            prefix=self.prefix + namespace + ".",
            cache=self.cache,
            src_to_idx=self.src_to_idx,
            record_src=self.record_src,
            record_functions=self.record_functions,
        )

    def get(self, key: str):
        key = self.prefix + key
        return self.cache[key]
