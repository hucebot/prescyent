"""Class and methods for a collection of
   sine waves. Each trajectory is delayed in time.
   Inspired by: https://github.com/pytorch/examples/tree/main/time_sequence_prediction
"""
from typing import Union, Dict
from pathlib import Path

import numpy as np
import torch

from prescyent.dataset.trajectories import Trajectory
from prescyent.dataset.dataset import MotionDataset, Trajectories
from prescyent.dataset.sine.config import DatasetConfig


class Dataset(MotionDataset):
    """Dataset is not splitted into test / train / val
    It as to be at initialisation, througt the parameters
    """

    DATASET_NAME = "Sine"

    def __init__(self, config: Union[Dict, DatasetConfig, Path, str] = None) -> None:
        self._init_from_config(config, DatasetConfig)
        self.trajectories = self._gen_data(
            self.config.length,
            self.config.period,
            int(self.config.size * self.config.ratio_train),
            int(self.config.size * self.config.ratio_test),
            int(self.config.size * self.config.ratio_val),
        )
        super().__init__(self.DATASET_NAME)

    def _gen_data(
        self, length: int, period: int, num_train: int, num_test: int, num_val: int
    ) -> Trajectories:
        rng = np.random.default_rng(42)
        train_trajectories = [
            self._gen_sine_wave(length, period, rng) for i in range(num_train)
        ]
        test_trajectories = [
            self._gen_sine_wave(length, period, rng) for i in range(num_test)
        ]
        val_trajectories = [
            self._gen_sine_wave(length, period, rng) for i in range(num_val)
        ]
        return Trajectories(train_trajectories, test_trajectories, val_trajectories)

    def _gen_sine_wave(self, length: int, period: int, rng: int) -> Trajectory:
        x = np.array(range(length)) + rng.integers(-4 * period, 4 * period)
        return Trajectory(
            torch.from_numpy(np.sin(x / 1.0 / period).astype("float64")).reshape(
                length, 1, 1
            ),
            100,
            "",
            "sin trajectory",
            ["sin(x)"],
        )
