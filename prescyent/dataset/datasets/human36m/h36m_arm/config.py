"""Pydantic config for H36M Arm dataset"""
from enum import Enum
from typing import List

from pydantic import model_validator

from prescyent.dataset.datasets.human36m.config import (
    DatasetConfig as H36MDatasetConfig,
)


class Arms(str, Enum):
    LEFT = "left"
    RIGHT = "right"


class DatasetConfig(H36MDatasetConfig):
    """Pydantic Basemodel for Dataset configuration"""

    bimanual: bool = True  # If bimanual, subsample dataset to both arms,
    # else we use the main arm:
    main_arm: Arms = Arms.RIGHT  # For mono arm, decide which is main arm
    in_points: List[int] = None
    out_points: List[int] = None

    @model_validator(mode="after")
    def check_out_points(self):
        """sets default value for out_points if None"""
        if self.out_points is None:
            if self.bimanual:
                self.out_points = list(range(14))
            else:
                self.out_points = list(range(7))
        return self

    @model_validator(mode="after")
    def check_in_points(self):
        """sets default value for in_points if None"""
        if self.in_points is None:
            if self.bimanual:
                self.in_points = list(range(14))
            else:
                self.in_points = list(range(7))
        return self
