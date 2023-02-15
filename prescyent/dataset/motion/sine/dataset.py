"""Class and methods for a collection of
   sine waves. Each trajectory is delayed in time.
   Inspired by: https://github.com/pytorch/examples/tree/main/time_sequence_prediction
"""
from typing import Callable, List, Union, Dict

import numpy as np
import torch

from prescyent.dataset.motion.episodes import Episode
from prescyent.utils.logger import logger, DATASET
from prescyent.dataset.motion.dataset import MotionDataset, Episodes
from prescyent.utils.dataset_manipulation import split_array_with_ratios
from prescyent.dataset.motion.sine.config import SineDatasetConfig

class SineDataset(MotionDataset):
    """TODO: present the dataset here
    Architecture

    Dataset is not splitted into test / train / val
    It as to be at initialisation, througt the parameters
    """
    def __init__(self, config: Union[Dict, SineDatasetConfig] = None,
                 scaler: Callable = None):
        if not config:
            config = SineDatasetConfig()
        self._init_from_config(config)
        self.feature_size = 1
        self.episodes = self._gen_data(self.config.length, self.config.period, 
            int(self.config.size * self.config.ratio_train), 
            int(self.config.size * self.config.ratio_test),
            int(self.config.size * self.config.ratio_val))
        super().__init__(scaler)

    def _init_from_config(self, config):
        if isinstance(config, dict):
            config = SineDatasetConfig(**config)
        self.config = config
        self.input_size = config.input_size
        self.output_size = config.output_size
        self.batch_size = config.batch_size

    def _gen_data(self, length, period, num_train, num_test, num_val):
        rng = np.random.default_rng(42)
        train_trajectories = [self._gen_sine_wave(length, period, rng) for i in range(num_train)]
        test_trajectories = [self._gen_sine_wave(length, period, rng) for i in range(num_test)]
        val_trajectories = [self._gen_sine_wave(length, period, rng) for i in range(num_val)]
        return Episodes(train_trajectories, test_trajectories, val_trajectories)

    def _gen_sine_wave(self, length, period, rng):
        x = np.array(range(length)) + rng.integers(-4 * period, 4 * period)
        return Episode(torch.Tensor(np.sin(x / 1.0 / period).astype('float64')).reshape(length, 1), 'TODO', ['sin(x)'])
