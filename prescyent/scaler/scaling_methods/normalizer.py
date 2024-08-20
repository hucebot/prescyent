import torch

# TODO: train, scale, uscale


class Normalizer:
    dim: int
    min_t: torch.Tensor
    max_t: torch.Tensor

    def __init__(self, dim: int) -> None:
        self.dim = dim

    def train(self, dataset_tensor: torch.Tensor):
        raise NotImplementedError("TODO")

    def scale(self, sample_tensor: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("TODO")

    def unscale(self, sample_tensor: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("TODO")
