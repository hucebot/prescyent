"""Feature for rotations"""
from typing import List

from prescyent.dataset.features.feature import Feature


class Rotation(Feature):
    """base class used for conversion"""

    def _is_convertible(self, __value: object) -> bool:
        return isinstance(__value, Rotation)


class RotationEuler(Rotation):
    """euler roll pitch yaw representation"""

    @property
    def num_dims(self) -> int:
        return 3

    @property
    def dims_names(self) -> List[str]:
        return ["roll", "pitch", "yaw"]


class RotationQuat(Rotation):
    """quaternion x, y, z, w representation"""

    @property
    def num_dims(self) -> int:
        return 4

    @property
    def dims_names(self) -> List[str]:
        return ["qx", "qy", "qz", "qw"]


class RotationRep6D(Rotation):
    """Continuous minimal representation from the rotmatrix, from:
    Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H. (2020).
    On the continuity of rotation representations in neural networks.
    arXiv preprint arXiv:1812.07035."""

    @property
    def num_dims(self) -> int:
        return 6

    @property
    def dims_names(self) -> List[str]:
        return ["x1", "x2", "x3", "y1", "y2", "y3"]


class RotationRotMat(Rotation):
    """rotation matrix"""

    @property
    def num_dims(self) -> int:
        return 9

    @property
    def dims_names(self) -> List[str]:
        return ["x1", "x2", "x3", "y1", "y2", "y3", "z1", "z2", "z3"]
