"""Common config elements for motion datasets usage"""
import random
from pathlib import Path
from typing import List, Optional

from pydantic import model_validator

import prescyent.dataset.features as tensor_features
from prescyent.base_config import BaseConfig
from prescyent.utils.enums import LearningTypes


class TrajectoriesDatasetConfig(BaseConfig):
    """Pydantic Basemodel for TrajectoriesDatasets configuration"""

    name: Optional[str] = None
    """Name of your dataset. WARNING, If you override default value, AutoDataset won't be able to load your dataset"""
    seed: int = None
    """A seed for all random operations in the dataset class"""

    # Dataloader values
    batch_size: int = 128
    """Size of the batch of all dataloaders"""
    num_workers: int = 1
    """See https://pytorch.org/docs/stable/data.html#single-and-multi-process-data-loading"""
    persistent_workers: bool = True
    """See https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader"""
    pin_memory: bool = True
    """See https://pytorch.org/docs/stable/data.html#memory-pinning"""
    save_samples_on_disk: bool = True
    """If True we'll use a tmp hdf5 file to store the x, y pairs and win some time at computation during training in the detriment of some init time and temporary disk space"""

    # x, y pairs related variables for motion data samples:
    learning_type: LearningTypes = LearningTypes.SEQ2SEQ
    """Method used to generate TrajectoryDataSamples"""
    frequency: int
    """The frequency in Hz of the dataset, If different from original data we'll use linear upsampling or downsampling of the data"""
    history_size: int
    """Number of timesteps as input"""
    future_size: int
    """Number of timesteps predicted as output"""
    in_features: Optional[tensor_features.Features]
    """List of features used as input, if None, use default from the dataset"""
    out_features: Optional[tensor_features.Features]
    """List of features used as output, if None, use default from the dataset"""
    in_points: Optional[List[int]]
    """Ids of the points used as input."""
    out_points: Optional[List[int]]
    """Ids of the points used as output."""
    context_keys: List[str] = []
    """List of the key of the tensors we'll pass as context to the predictor. Must be a subset of the existing context keys in the Dataset's Trajectories"""
    convert_trajectories_beforehand: bool = True
    """If in_features and out_features allows it, convert the trajectories as a preprocessing instead of in the dataloaders"""
    loop_over_traj: bool = False
    """Make the trajectory loop over itself where generating training pairs"""
    reverse_pair_ratio: float = 0
    """Do data augmentation by reversing some trajectories' sequence with given ratio as chance of occuring between 0 and 1"""

    @property
    def num_out_features(self) -> int:
        """number of different output features"""
        if self.out_features is None:
            return 0
        return len(self.out_features)

    @property
    def num_in_features(self) -> int:
        """number of different input features"""
        if self.in_features is None:
            return 0
        return len(self.in_features)

    @property
    def num_out_dims(self) -> int:
        """sum of the dims of output features"""
        if self.out_features is None:
            return 0
        return sum([len(feat.ids) for feat in self.out_features])

    @property
    def num_in_dims(self) -> int:
        """sum of the dims of input features"""
        if self.in_features is None:
            return 0
        return sum([len(feat.ids) for feat in self.in_features])

    @property
    def num_out_points(self) -> int:
        """number of output points"""
        if self.out_points is None:
            return 0
        return len(self.out_points)

    @property
    def num_in_points(self) -> int:
        """number of input points"""
        if self.in_points is None:
            return 0
        return len(self.in_points)

    @property
    def out_dims(self) -> List[int]:
        """list of dims sizes per output feature"""
        dims = []
        if self.out_features is None:
            return dims
        for feature in self.out_features:
            dims += feature.ids
        return dims

    @property
    def in_dims(self) -> List[int]:
        """list of dims sizes per input feature"""
        dims = []
        if self.in_features is None:
            return dims
        for feature in self.in_features:
            dims += feature.ids
        return dims

    @model_validator(mode="before")
    def unserialize_features(self):
        """turns features dict from json data into the Features object"""
        if self.get("out_features", None):
            if isinstance(self["out_features"], tensor_features.Feature):
                self["out_features"] = tensor_features.Features(
                    [self["out_features"]], index_name=False
                )
            if not isinstance(self["out_features"][0], tensor_features.Feature):
                self["out_features"] = tensor_features.Features(
                    [
                        getattr(tensor_features, feature["feature_class"])(
                            feature["ids"], name=feature["name"]
                        )
                        for feature in self["out_features"]
                    ],
                    index_name=False,
                )
            if isinstance(self["out_features"], list):
                self["out_features"] = tensor_features.Features(
                    self["out_features"], index_name=False
                )
        if self.get("in_features", None):
            if isinstance(self["in_features"], tensor_features.Feature):
                self["in_features"] = tensor_features.Features(
                    [self["in_features"]], index_name=False
                )
            if not isinstance(self["in_features"][0], tensor_features.Feature):
                self["in_features"] = tensor_features.Features(
                    [
                        getattr(tensor_features, feature["feature_class"])(
                            feature["ids"], name=feature["name"]
                        )
                        for feature in self["in_features"]
                    ],
                    index_name=False,
                )
            if isinstance(self["in_features"], list):
                self["in_features"] = tensor_features.Features(
                    self["in_features"], index_name=False
                )
        return self

    @model_validator(mode="after")
    def generate_random_seed_if_none(self):
        """generates a random seed if it is None"""
        if self.seed is None:
            self.seed = random.randint(1, 10**9)
        return self
