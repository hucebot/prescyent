"""Common config elements for motion datasets usage"""
from typing import Optional, List

from prescyent.dataset.config import MotionDatasetConfig
from prescyent.utils.enums import RotationRepresentation


class Dataset3dConfig(MotionDatasetConfig):
    """Pydantic Basemodel for MotionDatasets configuration"""

    coordinates_in: Optional[List[int]] = None  # None means all coordinates, x,y,z
    coordinates_out: Optional[List[int]] = None  # None means all coordinates, x,y,z
    rotation_representation_in: Optional[
        RotationRepresentation
    ]  # None means no rotation
    rotation_representation_out: Optional[
        RotationRepresentation
    ]  # None means no rotation
