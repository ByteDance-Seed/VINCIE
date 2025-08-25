 # Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
 # SPDX-License-Identifier: Apache-2.0 

"""
Utility functions for creating schedules and samplers from config.
"""

import torch
from omegaconf import DictConfig

from .samplers.base import Sampler
from .samplers.euler import EulerSampler
from .schedules.base import Schedule
from .schedules.lerp import LinearInterpolationSchedule
from .timesteps.base import SamplingTimesteps, TrainingTimesteps
from .timesteps.sampling.trailing import UniformTrailingSamplingTimesteps
from .timesteps.training.logitnormal import LogitNormalTrainingTimesteps


def create_schedule_from_config(
    config: DictConfig,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Schedule:
    """
    Create a schedule from configuration.
    """

    if config.type == "lerp":
        return LinearInterpolationSchedule(T=config.get("T", 1.0))

    raise NotImplementedError


def create_sampler_from_config(
    config: DictConfig,
    schedule: Schedule,
    timesteps: SamplingTimesteps,
) -> Sampler:
    """
    Create a sampler from configuration.
    """
    if config.type == "euler":
        return EulerSampler(
            schedule=schedule,
            timesteps=timesteps,
            prediction_type=config.prediction_type,
        )
    raise NotImplementedError


def create_training_timesteps_from_config(
    config: DictConfig,
    schedule: Schedule,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> TrainingTimesteps:
    if config.type == "logitnormal":
        return LogitNormalTrainingTimesteps(
            T=schedule.T,
            loc=config.loc,
            scale=config.scale,
        )
    raise NotImplementedError


def create_sampling_timesteps_from_config(
    config: DictConfig,
    schedule: Schedule,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> SamplingTimesteps:
    if config.type == "uniform_trailing":
        return UniformTrailingSamplingTimesteps(
            T=schedule.T,
            steps=config.steps,
            shift=config.get("shift", 1.0),
            device=device,
        )

