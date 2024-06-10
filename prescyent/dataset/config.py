"""Common config elements for motion datasets usage"""
import random
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, model_validator

import prescyent.dataset.features as tensor_features
from prescyent.utils.enums import LearningTypes


root_dir = Path(__file__).parent.parent.parent
DEFAULT_DATA_PATH = str(root_dir / "data" / "datasets")


class MotionDatasetConfig(BaseModel):
    """Pydantic Basemodel for MotionDatasets configuration"""

    batch_size: int = 128
    learning_type: LearningTypes = LearningTypes.SEQ2SEQ
    shuffle_train: bool = True
    shuffle_test: bool = False
    shuffle_val: bool = False
    num_workers: int = 0
    drop_last: bool = True
    persistent_workers: bool = False
    pin_memory: bool = True
    # x, y pairs related variables for motion data samples:
    history_size: int  # number of timesteps as input
    future_size: int  # number of predicted timesteps
    in_features: Optional[List[tensor_features.Feature]]
    out_features: Optional[List[tensor_features.Feature]]
    # do not mistake theses with the "used joint" one that is used on Trajectory level. Theses values are relative to the used_joints one
    in_points: Optional[List[int]]
    out_points: Optional[List[int]]
    convert_trajectories_beforehand: bool = True  # If in_features and out_features allows it, convert the trajectories as a preprocessing instead of in the dataloaders
    loop_over_traj: bool = False  # make the trajectory loop over itself where generating training pairs
    reverse_pair_ratio: float = 0  # Do data augmentation by reversing some trajectories' sequence with given ratio as chance of occuring between 0 and 1
    seed: int = None  # A seed for all random operations in the dataset class

    @property
    def num_out_features(self) -> int:
        if self.out_features is None:
            return 0
        return len(self.out_features)

    @property
    def num_in_features(self) -> int:
        if self.in_features is None:
            return 0
        return len(self.in_features)

    @property
    def num_out_dims(self) -> int:
        if self.out_features is None:
            return 0
        return sum([len(feat.ids) for feat in self.out_features])

    @property
    def num_in_dims(self) -> int:
        if self.in_features is None:
            return 0
        return sum([len(feat.ids) for feat in self.in_features])

    @property
    def num_out_points(self) -> int:
        if self.out_points is None:
            return 0
        return len(self.out_points)

    @property
    def num_in_points(self) -> int:
        if self.in_points is None:
            return 0
        return len(self.in_points)

    @property
    def out_dims(self) -> List[int]:
        dims = []
        if self.out_features is None:
            return dims
        for feature in self.out_features:
            dims += feature.ids
        return dims

    @property
    def in_dims(self) -> List[int]:
        dims = []
        if self.in_features is None:
            return dims
        for feature in self.in_features:
            dims += feature.ids
        return dims

    @model_validator(mode="before")
    def unserialize_features(self):
        if self.get("out_features", None):
            if isinstance(self["out_features"], tensor_features.Feature):
                self["out_features"] = [self["out_features"]]
            if not isinstance(self["out_features"][0], tensor_features.Feature):
                self["out_features"] = [
                    getattr(tensor_features, feature["name"])(feature["ids"])
                    for feature in self["out_features"]
                ]
        if self.get("in_features", None):
            if isinstance(self["in_features"], tensor_features.Feature):
                self["in_features"] = [self["in_features"]]
            if not isinstance(self["in_features"][0], tensor_features.Feature):
                self["in_features"] = [
                    getattr(tensor_features, feature["name"])(feature["ids"])
                    for feature in self["in_features"]
                ]
        return self


    @model_validator(mode="after")
    def generate_random_seed_if_none(self):
        if self.seed is None:
            self.seed = random.randint(1, 10**9)
        return self


    class Config:
        arbitrary_types_allowed = True
