"""class for normalization of tensors"""
import json
from typing import List, Optional

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from prescyent.utils.enums import TrajectoryDimensions


class Normalizer:
    """class to perfom normalization as a scaling method"""

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
        for batch in tqdm(
            dataset_dataloader, desc="iterating over dataset", colour="red"
        ):
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

        min_t = self.min_t.detach().clone().to(sample_tensor.device)
        max_t = self.max_t.detach().clone().to(sample_tensor.device)
        if self.dim == [0, 1, 3]:
            sample_tensor = sample_tensor.transpose(2, 3)
        if point_ids and 2 not in self.dim:
            # If we scale a subset of the dataset feats, and it is not averaged over
            min_t = min_t[point_ids]
            max_t = max_t[point_ids]
        if feat_ids and 3 not in self.dim:
            # If we scale a subset of the dataset feats, and it is not averaged over
            min_t = min_t[..., feat_ids]
            max_t = max_t[..., feat_ids]
        res = (sample_tensor - min_t) / (max_t - min_t + self.eps)
        if self.dim == [0, 1, 3]:
            res = res.transpose(2, 3)
        return res

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

        min_t = self.min_t.detach().clone().to(sample_tensor.device)
        max_t = self.max_t.detach().clone().to(sample_tensor.device)
        if self.dim == [0, 1, 3]:
            sample_tensor = sample_tensor.transpose(2, 3)
        if point_ids and 2 not in self.dim:
            # If we unscale a subset of the dataset feats, and it is not averaged over
            min_t = min_t[point_ids]
            max_t = max_t[point_ids]
        if feat_ids and 3 not in self.dim:
            # If we unscale a subset of the dataset feats, and it is not averaged over
            min_t = min_t[..., feat_ids]
            max_t = max_t[..., feat_ids]
        res = sample_tensor * (max_t - min_t) + min_t
        if self.dim == [0, 1, 3]:
            res = res.transpose(2, 3)
        return res
