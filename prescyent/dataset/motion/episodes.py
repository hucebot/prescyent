from typing import Callable, List

import torch


class Episode():
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
    def train_scaled(self):
        return [episode.scaled_tensor for episode in self.train]

    @property
    def test_scaled(self):
        return [episode.scaled_tensor for episode in self.test]

    @property
    def val_scaled(self):
        return [episode.scaled_tensor for episode in self.val]

    def _all_len(self):
        return len(self.train) + len(self.test) + len(self.val)
