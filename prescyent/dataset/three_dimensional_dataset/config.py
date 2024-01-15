"""Common config elements for motion datasets usage"""
from typing import Optional, List

from prescyent.dataset.config import MotionDatasetConfig
from prescyent.utils.enums import RotationRepresentation
from prescyent.utils.torch_rotation import rotrep2size


class Dataset3dConfig(MotionDatasetConfig):
    """Pydantic Basemodel for MotionDatasets configuration"""

    coordinates_in: List[int] = [0, 1, 2]  # Means all coordinates: x, y, z
    coordinates_out: List[int] = [0, 1, 2]  # Means all coordinates: x, y, z
    rotation_representation_in: Optional[
        RotationRepresentation
    ]  # None means no rotation
    rotation_representation_out: Optional[
        RotationRepresentation
    ]  # None means no rotation

    @property
    def num_out_dims(self):
        return len(self.coordinates_out) + rotrep2size(self.rotation_representation_out)

    @property
    def num_in_dims(self):
        return len(self.coordinates_in) + rotrep2size(self.rotation_representation_in)
