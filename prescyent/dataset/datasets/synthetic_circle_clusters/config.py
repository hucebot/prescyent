"""Config elements for SCC dataset usage"""
from typing import List

from pydantic import model_validator, field_validator

from prescyent.dataset.config import TrajectoriesDatasetConfig
from prescyent.dataset.features import Features
from .metadata import DEFAULT_FEATURES, POINT_LABELS


class SCCDatasetConfig(TrajectoriesDatasetConfig):
    """Pydantic Basemodel for SCCDataset configuration"""

    ratio_train: float = 0.7
    """ratio of trajectories placed in Trajectories.train"""
    ratio_test: float = 0.15
    """ratio of trajectories placed in Trajectories.test"""
    ratio_val: float = 0.15
    """ratio of trajectories placed in Trajectories.val"""
    # circle parameters
    num_trajs: List[int] = [25, 25]
    """Number of trajectory generated per cluster"""
    starting_xs: List[float] = [0, 4]
    """x coordinate for each cluster"""
    starting_ys: List[float] = [0, 0]
    """y coordinate for each cluster"""
    radius: List[float] = [1, 1]
    """radius for each cluster"""
    radius_eps: float = 0.01
    """variation for radius"""
    perturbation_range: float = 0.1
    """perturbation over the shape's main points"""
    num_perturbation_points: int = 10
    """number of perturbation points"""
    num_points: int = 100
    """number of points in the final shape after smoothing"""
    # Override default values with the dataset's
    frequency: int = 10
    """The frequency in Hz of the dataset, If different from original data we'll use linear upsampling or downsampling of the data"""
    history_size: int = 10
    """Number of timesteps as input"""
    future_size: int = 10
    """Number of timesteps predicted as output"""
    in_features: Features = DEFAULT_FEATURES
    """List of features used as input, if None, use default from the dataset"""
    out_features: Features = DEFAULT_FEATURES
    """List of features used as output, if None, use default from the dataset"""
    in_points: List[int] = list(range(len(POINT_LABELS)))
    """Ids of the points used as input."""
    out_points: List[int] = list(range(len(POINT_LABELS)))
    """Ids of the points used as output."""

    @property
    def num_clusters(self) -> int:
        return len(self.num_trajs)

    @property
    def num_traj(self) -> int:
        return sum(self.num_trajs)

    @model_validator(mode="after")
    def check_list_size_matches(self):
        """
        Then lenght of num_trajs, starting_xs, starting_ys and radius
        must mmatch because they are used for each cluster
        """
        if (
            self.num_clusters != len(self.starting_xs)
            or self.num_clusters != len(self.starting_ys)
            or self.num_clusters != len(self.radius)
        ):
            raise ValueError(
                '"num_trajs", "starting_xs", "starting_ys" and "radius" must be lists of same size.'
            )
        return self

    @field_validator("context_keys")
    def check_context_keys(cls, value):
        """check that requested keys exists in the dataset"""
        if value:
            raise ValueError("This dataset cannot handle context keys")
        return value
