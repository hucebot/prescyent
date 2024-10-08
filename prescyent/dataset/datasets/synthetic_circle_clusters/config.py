"""Config elements for SCC dataset usage"""
from typing import List

from pydantic import model_validator, ValidationError

from prescyent.dataset.config import MotionDatasetConfig
from prescyent.dataset.features import Features
from .metadata import DEFAULT_FEATURES, POINT_LABELS


class DatasetConfig(MotionDatasetConfig):
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
    history_size: int = 10
    future_size: int = 10
    in_features: Features = DEFAULT_FEATURES
    out_features: Features = DEFAULT_FEATURES
    in_points: List[int] = list(range(len(POINT_LABELS)))
    out_points: List[int] = list(range(len(POINT_LABELS)))

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

    @model_validator(mode="after")
    def check_context_keys(self):
        """check that requested keys exists in the dataset"""
        if self.context_keys:
            raise ValidationError("This dataset cannot handle context keys")
        return self
