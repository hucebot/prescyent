"""Module for trajectories classes"""
from pathlib import Path
from typing import List

import torch


class Trajectory:
    """
    An trajectory represents a full dataset sample, that we can retrieve with its file name
    An trajectory tracks n dimensions in time, represented in a tensor of shape (seq_len, n_dim)
    """

    tensor: torch.Tensor
    frequency: int
    file_path: str
    title: str
    dimension_names: List[str]

    def __init__(
        self,
        tensor: torch.Tensor,
        frequency: int,
        file_path: str = "trajectory_file_path",
        title: str = "trajectory_name",
        dimension_names: List[str] = ["y_infos"],
    ) -> None:
        self.tensor = tensor
        self.frequency = frequency
        self.file_path = file_path
        self.title = title
        self.dimension_names = dimension_names

    def __getitem__(self, index):
        return self.tensor[index]

    def __len__(self):
        return len(self.tensor)

    def __str__(self) -> str:
        return self.title

    @property
    def shape(self):
        return self.tensor.shape
