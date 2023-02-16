"""Module for trajectories classes"""
from typing import Callable, List

import torch


class Trajectory():
    """
    An trajectory represents a full dataset sample, that we can retrieve with its file name
    An trajectory tracks n dimensions in time, represented in a tensor of shape (seq_len, n_dim)
    We also store the scaled tensor, for interactions with the models
    """
    tensor: torch.Tensor
    scaled_tensor: torch.Tensor
    file_path: str
    dimension_names: List[str]

    def __init__(self, tensor: torch.Tensor,
                 file_path: str = "trajectory_name",
                 dimension_names: List[str] = ["y_infos"],
                 scaled_tensor: torch.Tensor = None) -> None:
        self.tensor = tensor
        self.scaled_tensor = scaled_tensor
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
    _scale_function: Callable

    def __init__(self, train: List[Trajectory],
                 test: List[Trajectory], val: List[Trajectory]) -> None:
        self.train = train
        self.test = test
        self.val = val

    @property
    def scale_function(self):
        return self._scale_function

    @scale_function.setter
    def scale_function(self, scale_function):
        """when setting a scaler, creating the scaled tensor for each trajectories"""
        self._scale_function = scale_function
        self._scale_tensors()

    def _scale_tensors(self):
        for trajectory in self.train:
            trajectory.scaled_tensor = self.scale_function(trajectory.tensor)
        for trajectory in self.test:
            trajectory.scaled_tensor = self.scale_function(trajectory.tensor)
        for trajectory in self.val:
            trajectory.scaled_tensor = self.scale_function(trajectory.tensor)

    @property
    def train_scaled(self) -> List[torch.Tensor]:
        return [trajectory.scaled_tensor for trajectory in self.train]

    @property
    def test_scaled(self) -> List[torch.Tensor]:
        return [trajectory.scaled_tensor for trajectory in self.test]

    @property
    def val_scaled(self) -> List[torch.Tensor]:
        return [trajectory.scaled_tensor for trajectory in self.val]

    def _all_len(self):
        return len(self.train) + len(self.test) + len(self.val)
