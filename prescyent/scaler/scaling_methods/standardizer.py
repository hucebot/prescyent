import json
import math
from typing import List, Optional

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from prescyent.utils.enums import TrajectoryDimensions


class Standardizer:
    """class to perform standardization as a scaling method"""

    mean: torch.Tensor
    std: torch.Tensor
    dim: List[int]
    eps: float

    def __init__(self, scaling_axis: TrajectoryDimensions, eps=1e-8) -> None:
        self.dim = list(
            set(json.loads(scaling_axis.value) + [0, 1])
        )  # + [0, 1] because we scale features, so we at least compute over batch and time_frames
        self.eps = eps

    def train(
        self, dataset_dataloader: DataLoader, feat_ids: Optional[List[int]] = None
    ):
        """Train the standardizer based on a dataloader over the whole dataset and the dataset's features

        Args:
            dataset_dataloader (DataLoader): Dataloader over the whole training dataset
            feat_ids (Optional[List[int]], optional): List of the dataset features. Defaults to None.
        """

        n_samples = 0
        total_sum = 0
        total_sum_sq = 0
        for batch in tqdm(
            dataset_dataloader, desc="iterating over dataset", colour="red"
        ):
            data = batch.unsqueeze(0)
            if feat_ids:
                data = data[..., feat_ids]
            sample_size = math.prod([data.size(dim) for dim in self.dim])
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
        """Standardizes input tensor

        Args:
            sample_tensor (torch.Tensor): input tensor
            point_ids (List[int], optional): list of point ids to standardize on.
                If None, standardize over whole given points. Defaults to None.
            feat_ids (List[int], optional): list of feature ids to standardize on.
                If None, standardize over whole given features. Defaults to None.

        Returns:
            torch.Tensor: Standardized input tensor
        """

        std = self.std.detach().clone().to(sample_tensor.device)
        mean = self.mean.detach().clone().to(sample_tensor.device)
        if self.dim == [0, 1, 3]:
            sample_tensor = sample_tensor.transpose(2, 3)
        if point_ids and 2 not in self.dim:
            # If we unscale a subset of the dataset feats, and it is not averaged over
            std = std[point_ids]
            mean = mean[point_ids]
        if feat_ids and 3 not in self.dim:
            # If we unscale a subset of the dataset feats, and it is not averaged over
            std = std[..., feat_ids]
            mean = mean[..., feat_ids]
        res = (sample_tensor - mean) / (std + self.eps)
        if self.dim == [0, 1, 3]:
            res = res.transpose(2, 3)
        return res

    def unscale(
        self,
        sample_tensor: torch.Tensor,
        point_ids: List[int] = None,
        feat_ids: List[int] = None,
    ) -> torch.Tensor:
        """Unstandardizes input tensor

        Args:
            sample_tensor (torch.Tensor): input tensor
            point_ids (List[int], optional): list of point ids to Unstandardize on.
                If None, Unstandardize over whole given points. Defaults to None.
            feat_ids (List[int], optional): list of feature ids to Unstandardize on.
                If None, Unstandardize over whole given features. Defaults to None.

        Returns:
            torch.Tensor: Unstandardized input tensor
        """

        std = self.std.detach().clone().to(sample_tensor.device)
        mean = self.mean.detach().clone().to(sample_tensor.device)
        if self.dim == [0, 1, 3]:
            sample_tensor = sample_tensor.transpose(2, 3)
        if point_ids and 2 not in self.dim:
            # If we unscale a subset of the dataset feats, and it is not averaged over
            std = std[point_ids]
            mean = mean[point_ids]
        if feat_ids and 3 not in self.dim:
            # If we unscale a subset of the dataset feats, and it is not averaged over
            std = std[..., feat_ids]
            mean = mean[..., feat_ids]
        res = sample_tensor * std + mean
        if self.dim == [0, 1, 3]:
            res = res.transpose(2, 3)
        return res
