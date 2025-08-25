 # Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
 # SPDX-License-Identifier: Apache-2.0 

from typing import Optional
from omegaconf import OmegaConf

from common.entrypoint import Entrypoint

from ..fs import set_download_dir
from .dataclass import PersistedTrainingState
from .manager import PersistenceManager
from .utils import set_local_dir


class PersistenceMixin:
    """
    Provide persistence capability.
    Config must contain a "persistence" key.
    """

    # ----------------- Example Config -----------------
    # persistence:
    #   path: hdfs://path/to/location  (required)
    #   override: False                (default False)
    # --------------------------------------------------

    persistence: PersistenceManager
    resume: Optional[PersistedTrainingState]

    def configure_persistence(
        self,
        mode: str = "train",
        models: list = [],
        optimizers: list = [],
    ):
        """
        Configure persistence and resume.

        Args:
            mode: "train" will resume, "eval" won't resume training states.
            models: list of models to check in the ckpt path.
            optimizers: list of optimizers to check in the ckpt path.
        """
        assert mode in ["train", "eval"]
        assert isinstance(self, Entrypoint)
        set_download_dir(self.config.persistence.get("download_dir", None))
        set_local_dir(self.config.persistence.get("local_dir", None))

        self.persistence = PersistenceManager(path=self.config.persistence.path)
        if mode == "train":
            assert NotImplementedError