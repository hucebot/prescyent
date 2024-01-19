from typing import List

from prescyent.dataset.features.feature import Feature


class Rotation(Feature):
    """"""
    def _is_convertible(self, __value: object) -> bool:
        return isinstance(__value, Rotation)


class RotationEuler(Rotation):
    """"""

    @property
    def num_dims(self) -> int:
        return 3

    @property
    def dims_names(self) -> List[str]:
        return ["roll", "pitch", "yaw"]


class RotationQuat(Rotation):
    """"""

    @property
    def num_dims(self) -> int:
        return 4


    @property
    def dims_names(self) -> List[str]:
        return ["qx", "qy", "qz", "qw"]


class RotationRep6D(Rotation):
    """"""

    @property
    def num_dims(self) -> int:
        return 6

    @property
    def dims_names(self) -> List[str]:
        return ["x1", "x2", "x3", "y1", "y2", "y3"]


class RotationRotMat(Rotation):
    """"""

    @property
    def num_dims(self) -> int:
        return 9

    @property
    def dims_names(self) -> List[str]:
        return ["x1", "x2", "x3", "y1", "y2", "y3", "z1", "z2", "z3"]
