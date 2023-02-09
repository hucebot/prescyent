"""Module for episodes classes"""
from typing import Callable, List

import torch


class Episode():
    """
    An episode represents a full dataset sample, that we can retreive with its file name
    An episode tracks n dimensions in time, represented in a tensor of shape (seq_len, n_dim)
    We also store the scaled tensor, for interactions with the models
    """
    tensor: torch.Tensor
    scaled_tensor: torch.Tensor = None
    file_path: str
    dimension_names: List[str]

    def __init__(self, tensor: torch.Tensor,
                 file_path: str, dimension_names: List[str]) -> None:
        self.tensor = tensor
        self.file_path = file_path
        self.dimension_names = dimension_names

    def __getitem__(self, index):
        return self.tensor[index]

    def __len__(self):
        return len(self.tensor)

    @property
    def shape(self):
        return self.tensor.shape


class Episodes():
    """Episodes are collections of Episodes, organized into train, val, test"""
    train: List[Episode]
    test: List[Episode]
    val: List[Episode]
    _scale_function: Callable

    def __init__(self, train: List[Episode],
                 test: List[Episode], val: List[Episode]) -> None:
        self.train = train
        self.test = test
        self.val = val

    @property
    def scale_function(self):
        return self._scale_function

    @scale_function.setter
    def scale_function(self, scale_function):
        """when setting a scaler, creating the """
        self._scale_function = scale_function
        for episode in self.train:
            episode.scaled_tensor = scale_function(episode.tensor)
        for episode in self.test:
            episode.scaled_tensor = scale_function(episode.tensor)
        for episode in self.val:
            episode.scaled_tensor = scale_function(episode.tensor)

    @property
    def train_scaled(self) -> List[torch.Tensor]:
        return [episode.scaled_tensor for episode in self.train]

    @property
    def test_scaled(self) -> List[torch.Tensor]:
        return [episode.scaled_tensor for episode in self.test]

    @property
    def val_scaled(self) -> List[torch.Tensor]:
        return [episode.scaled_tensor for episode in self.val]

    def _all_len(self):
        return len(self.train) + len(self.test) + len(self.val)
