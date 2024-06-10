"""Config elements for SCC dataset usage"""
from typing import List

from prescyent.dataset.config import MotionDatasetConfig, model_validator
from prescyent.dataset.features import Feature
from .metadata import FEATURES, POINT_LABELS


class DatasetConfig(MotionDatasetConfig):
    """Pydantic Basemodel for SCCDataset configuration"""

    history_size: int = 10
    future_size: int = 10
    subsampling_step: int = 1  # subsampling -> 10 Hz to 10Hz
    ratio_train: float = (
        0.7  # ratios used to sample the dataset into train, test and val groups
    )
    ratio_test: float = 0.15
    ratio_val: float = 0.15
    # circle parameters
    num_trajs: List[int] = [25, 25]  # Number of trajectory generated per cluster
    starting_xs: List[float] = [0, 4]  # x coordinate for each cluster
    starting_ys: List[float] = [0, 0]  # y coordinate for each cluster
    radius: List[float] = [1, 1]  # radius for each cluster
    radius_eps: float = 0.01  # variation for radius
    perturbation_range: float = 0.1  # perturbation over the shape's main points
    num_perturbation_points: int = 10  # number of perturbation points
    num_points: int = 100  # number of points in the final shape after smoothing
    in_features: List[Feature] = FEATURES
    out_features: List[Feature] = FEATURES
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
