"""Class and methods for a collection of
   sine waves. Each trajectory is delayed in time.
   Inspired by: https://github.com/pytorch/examples/tree/main/time_sequence_prediction
"""
from typing import Callable, Union, Dict

import numpy as np
import torch

from prescyent.dataset.trajectories import Trajectory
from prescyent.dataset.dataset import MotionDataset, Trajectories
from prescyent.dataset.sine.config import DatasetConfig


class Dataset(MotionDataset):
    """Dataset is not splitted into test / train / val
    It as to be at initialisation, througt the parameters
    """
    def __init__(self, config: Union[Dict, DatasetConfig] = None,
                 scaler: Callable = None):
        if not config:
            config = DatasetConfig()
        self._init_from_config(config)
        self.feature_size = 1
        self.trajectories = self._gen_data(self.config.length, self.config.period,
                                           int(self.config.size * self.config.ratio_train),
                                           int(self.config.size * self.config.ratio_test),
                                           int(self.config.size * self.config.ratio_val))
        super().__init__(scaler)

    def _init_from_config(self, config):
        if isinstance(config, dict):
            config = DatasetConfig(**config)
        self.config = config
        self.history_size = config.history_size
        self.future_size = config.future_size
        self.batch_size = config.batch_size

    def _gen_data(self, length, period, num_train, num_test, num_val):
        rng = np.random.default_rng(42)
        train_trajectories = [self._gen_sine_wave(length, period, rng) for i in range(num_train)]
        test_trajectories = [self._gen_sine_wave(length, period, rng) for i in range(num_test)]
        val_trajectories = [self._gen_sine_wave(length, period, rng) for i in range(num_val)]
        return Trajectories(train_trajectories, test_trajectories, val_trajectories)

    def _gen_sine_wave(self, length, period, rng):
        x = np.array(range(length)) + rng.integers(-4 * period, 4 * period)
        return Trajectory(
            torch.Tensor(np.sin(x / 1.0 / period).astype('float64')).reshape(length, 1),
            'sin trajectory',
            ['sin(x)'])
