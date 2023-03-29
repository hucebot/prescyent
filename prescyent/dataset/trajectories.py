"""Module for trajectories classes"""
from typing import Callable, List

import torch


class Trajectory():
    """
    An trajectory represents a full dataset sample, that we can retrieve with its file name
    An trajectory tracks n dimensions in time, represented in a tensor of shape (seq_len, n_dim)
    """
    tensor: torch.Tensor
    file_path: str
    dimension_names: List[str]

    def __init__(self, tensor: torch.Tensor,
                 file_path: str = "trajectory_name",
                 dimension_names: List[str] = ["y_infos"]) -> None:
        self.tensor = tensor
        self.file_path = file_path
        self.dimension_names = dimension_names

    def __getitem__(self, index):
        return self.tensor[index]

    def __len__(self):
        return len(self.tensor)

    def __str__(self) -> str:
        return self.file_path

    @property
    def shape(self):
        return self.tensor.shape


class Trajectories():
    """Trajectories are collections of n Trajectory, organized into train, val, test"""
    train: List[Trajectory]
    test: List[Trajectory]
    val: List[Trajectory]

    def __init__(self, train: List[Trajectory],
                 test: List[Trajectory], val: List[Trajectory]) -> None:
        self.train = train
        self.test = test
        self.val = val

    def _all_len(self):
        return len(self.train) + len(self.test) + len(self.val)
