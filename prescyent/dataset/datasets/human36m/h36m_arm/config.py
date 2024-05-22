"""Pydantic config for H36M Arm dataset"""
from enum import Enum
from typing import Optional, List

from prescyent.dataset.datasets.human36m.config import (
    DatasetConfig as H36MDatasetConfig,
)
from .metadata import LEFT_ARM_LABELS, RIGHT_ARM_LABELS


class Arms(str, Enum):
    LEFT = "left"
    RIGHT = "right"


class DatasetConfig(H36MDatasetConfig):
    """Pydantic Basemodel for Dataset configuration"""

    used_joints: List[int] = list(
        range(len(LEFT_ARM_LABELS + RIGHT_ARM_LABELS))
    )  # Use all joints from H36M by default
    bimanual: bool = True  # If bimanual, subsample dataset to both arms,
    # else we use the following:
    main_arm: Arms = None  # For mono arm, decide which is main arm
    use_both_arms: bool = False  # Can use the second arm to augment data
    in_points: List[int] = list(range(len(used_joints)))
    out_points: List[int] = list(range(len(used_joints)))
    # TODO
    # mirror_second_arm: bool = False   # For data augmentation mirror second arms
