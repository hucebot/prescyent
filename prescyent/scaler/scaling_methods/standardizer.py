import json
import math
from typing import List, Optional

import torch
from torch.utils.data import DataLoader

from prescyent.utils.enums import TrajectoryDimensions


class Standardizer:
    mean: torch.Tensor
    std: torch.Tensor
    dim: Optional[List[int]]
    eps: float

    def __init__(self, scaling_axis: Optional[TrajectoryDimensions], eps=1e-8) -> None:
        self.dim = list(
            set(json.loads(scaling_axis.value) + [0, 1])
        )  # Always scale on batch and time_frames
        self.eps = eps

    def train(
        self, dataset_dataloader: DataLoader, feat_ids: Optional[List[int]] = None
    ):
        n_samples = 0
        total_sum = 0
        total_sum_sq = 0
        for batch in dataset_dataloader:
            data = batch.unsqueeze(0)
            sample_size = math.prod([data.size(dim) for dim in self.dim])
            if feat_ids:
                data = data[..., feat_ids]
            n_samples += sample_size
            total_sum += data.sum(dim=self.dim)
            total_sum_sq += (data**2).sum(dim=self.dim)

        self.mean = total_sum / n_samples
        variance = (total_sum_sq / n_samples) - (self.mean**2)
        self.std = torch.sqrt(variance + self.eps)

    def scale(
        self,
        sample_tensor: torch.Tensor,
        point_ids: List[int] = None,
        feat_ids: List[int] = None,
    ) -> torch.Tensor:
        if self.dim == [0, 1, 3]:
            sample_tensor = sample_tensor.transpose(2, 3)
        if (
            point_ids and 2 not in self.dim and feat_ids and 3 not in self.dim
        ):  # If we scale a subset of the dataset points, and it is not averaged over
            res = (sample_tensor - self.mean[point_ids, feat_ids]) / (
                self.std[point_ids, feat_ids] + self.eps
            )
        elif (
            point_ids and 2 not in self.dim
        ):  # If we scale a subset of the dataset points, and it is not averaged over
            res = (sample_tensor - self.mean[point_ids]) / (
                self.std[point_ids] + self.eps
            )
        elif (
            feat_ids and 3 not in self.dim
        ):  # If we scale a subset of the dataset feats, and it is not averaged over
            res = (sample_tensor - self.mean[..., feat_ids]) / (
                self.std[..., feat_ids] + self.eps
            )
        else:
            res = (sample_tensor - self.mean) / (self.std + self.eps)
        if self.dim == [0, 1, 3]:
            res = res.transpose(2, 3)
        return res

    def unscale(
        self,
        sample_tensor: torch.Tensor,
        point_ids: List[int] = None,
        feat_ids: List[int] = None,
    ) -> torch.Tensor:
        if self.dim == [0, 1, 3]:
            sample_tensor = sample_tensor.transpose(2, 3)
        if (
            point_ids and 2 not in self.dim and feat_ids and 3 not in self.dim
        ):  # If we unscale a subset of the dataset points, and it is not averaged over
            res = (
                sample_tensor * self.std[point_ids, feat_ids]
                + self.mean[point_ids, feat_ids]
            )
        elif (
            point_ids and 2 not in self.dim
        ):  # If we unscale a subset of the dataset points, and it is not averaged over
            res = sample_tensor * self.std[point_ids] + self.mean[point_ids]
        elif (
            feat_ids and 3 not in self.dim
        ):  # If we unscale a subset of the dataset feats, and it is not averaged over
            res = sample_tensor * self.std[..., feat_ids] + self.mean[..., feat_ids]
        else:
            res = sample_tensor * self.std + self.mean
        if self.dim == [0, 1, 3]:
            res = res.transpose(2, 3)
        return res
