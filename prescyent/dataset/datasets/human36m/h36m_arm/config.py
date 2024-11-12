"""Pydantic config for H36M Arm dataset"""
from enum import Enum
from typing import List

from pydantic import model_validator

from prescyent.dataset.datasets.human36m.config import H36MDatasetConfig


class Arms(str, Enum):
    LEFT = "left"
    RIGHT = "right"


class H36MArmDatasetConfig(H36MDatasetConfig):
    """Pydantic Basemodel for Dataset configuration"""

    bimanual: bool = True
    """If bimanual, subsample dataset to both arms"""
    main_arm: Arms = Arms.RIGHT
    """If not bimanual, we use only the main arm"""
    in_points: List[int] = None
    """Ids of the points used as input."""
    out_points: List[int] = None
    """Ids of the points used as output."""

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
