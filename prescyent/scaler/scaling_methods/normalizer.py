import json
import math
from typing import List, Optional

import torch
from torch.utils.data import DataLoader

from prescyent.utils.enums import TrajectoryDimensions


class Normalizer:
    min_t: torch.Tensor
    max_t: torch.Tensor
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
        """Train the normalizer based on a dataloader over the whole dataset and the dataset's features

        Args:
            dataset_dataloader (DataLoader): Dataloader over the whole training dataset
            feat_ids (Optional[List[int]], optional): List of the dataset features. Defaults to None.
        """
        dataset_dataloader = iter(dataset_dataloader)
        batch = next(dataset_dataloader)
        data = batch.unsqueeze(0)
        if feat_ids:
            data = data[..., feat_ids]
        self.min_t = torch.amin(data, dim=self.dim)
        self.max_t = torch.amax(data, dim=self.dim)
        for batch in dataset_dataloader:
            data = batch.unsqueeze(0)
            if feat_ids:
                data = data[..., feat_ids]
            curr_min_t = torch.amin(data, dim=self.dim)
            curr_max_t = torch.amax(data, dim=self.dim)
            self.min_t = torch.amin(
                torch.cat([self.min_t.unsqueeze(0), curr_min_t.unsqueeze(0)], dim=0),
                dim=0,
            )
            self.max_t = torch.amax(
                torch.cat([self.max_t.unsqueeze(0), curr_max_t.unsqueeze(0)], dim=0),
                dim=0,
            )

    def scale(
        self,
        sample_tensor: torch.Tensor,
        point_ids: List[int] = None,
        feat_ids: List[int] = None,
    ) -> torch.Tensor:
        """Normalizes input tensor

        Args:
            sample_tensor (torch.Tensor): input tensor
            point_ids (List[int], optional): list of point ids to Normalize on.
                If None, Normalize over whole given points. Defaults to None.
            feat_ids (List[int], optional): list of feature ids to Normalize on.
                If None, Normalize over whole given features. Defaults to None.

        Returns:
            torch.Tensor: Normalized input tensor
        """
        return (sample_tensor - self.min_t) / (self.max_t - self.min_t + self.eps)

    def unscale(
        self,
        sample_tensor: torch.Tensor,
        point_ids: List[int] = None,
        feat_ids: List[int] = None,
    ) -> torch.Tensor:
        """Unnormalizes input tensor

        Args:
            sample_tensor (torch.Tensor): input tensor
            point_ids (List[int], optional): list of point ids to Unnormalize on.
                If None, Unnormalize over whole given points. Defaults to None.
            feat_ids (List[int], optional): list of feature ids to Unnormalize on.
                If None, Unnormalize over whole given features. Defaults to None.

        Returns:
            torch.Tensor: Unnormalized input tensor
        """
        return sample_tensor * (self.max_t - self.min_t) + self.min_t
