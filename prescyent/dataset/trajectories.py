"""Module for trajectories classes"""
from typing import List

import torch

from prescyent.dataset.trajectory import Trajectory


class Trajectories:
    """Trajectories are collections of n Trajectory, organized into train, val, test"""

    train: List[Trajectory]
    test: List[Trajectory]
    val: List[Trajectory]

    def __init__(
        self, train: List[Trajectory], test: List[Trajectory], val: List[Trajectory]
    ) -> None:
        self.train = train
        self.test = test
        self.val = val

    def _all_len(self):
        return len(self.train) + len(self.test) + len(self.val)

    def __len__(self):
        return self._all_len()

    def __getitem__(self, index):
        return self.train[index]
