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
    point_parents: List[int]
    dimension_names: List[str]

    def __init__(
        self,
        tensor: torch.Tensor,
        frequency: int,
        file_path: str = "trajectory_file_path",
        title: str = "trajectory_name",
        point_parents: List[int] = [-1],
        dimension_names: List[str] = ["y_infos"],
    ) -> None:
        self.tensor = tensor
        self.frequency = frequency
        self.file_path = file_path
        self.title = title
        self.point_parents = point_parents
        self.dimension_names = dimension_names

    def __getitem__(self, index) -> torch.Tensor:
        return self.tensor[index]

    def __len__(self) -> int:
        return len(self.tensor)

    def __str__(self) -> str:
        return self.title

    @property
    def shape(self) -> torch.Size:
        return self.tensor.shape

    @property
    def duration(self) -> float:
        """duration in seconds"""
        return len(self.tensor) / self.frequency
