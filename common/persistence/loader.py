 # Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
 # SPDX-License-Identifier: Apache-2.0 

"""
Model loader.
"""

from typing import Optional

from common.fs import exists

from .dataclass import PersistedModel
from .manager import PersistenceManager


def load_model_from_path(
    path: str,
    name: str = "model",
    step: Optional[int] = None,
) -> PersistedModel:
    """
    Load a persisted model from a given path and name.
    """
    manager = PersistenceManager(path)
    archive = manager.load_step(step)
    if archive is None:
        raise ValueError(f"Model not found under path: {path} and step: {step}")
    if name not in archive.models.keys():
        raise ValueError(f"Model name: {name} not found in archive.")
    return archive.models[name]


def load_model_from_task(
    task: str,
    name: str = "model",
    step: Optional[int] = None,
    distributed: Optional[bool] = True,
) -> PersistedModel:
    """
    Load a persisted model from merlin task id and name.
    Task id can be found as $MERLIN_JOB_ID on the machine.
    Example merlin task id: "d5ad7ed1c6acd81f"
    """
    task_info = PersistenceManager.get_task(task)
    if not exists(task_info.system.path):
        raise ValueError(f"Task not found: {task}")
    return load_model_from_path(
        path=task_info.system.load(distributed=distributed).path,
        name=name,
        step=step,
    )
