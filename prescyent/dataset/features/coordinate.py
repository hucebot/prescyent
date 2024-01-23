"""Feature used to represent coordinates from 1D to 3D"""
from typing import List

from prescyent.dataset.features.feature import Feature


class Coordinate(Feature):
    """parent class for coordinates, used for conversion"""

    def _is_convertible(self, __value: object) -> bool:
        return isinstance(__value, Coordinate) and len(self.ids) > len(__value.ids)


class CoordinateX(Coordinate):
    """1D"""

    @property
    def num_dims(self) -> int:
        return 1

    @property
    def dims_names(self) -> List[str]:
        return ["x"]


class CoordinateXY(Coordinate):
    """2D"""

    @property
    def num_dims(self) -> int:
        return 2

    @property
    def dims_names(self) -> List[str]:
        return ["x", "y"]


class CoordinateXYZ(Coordinate):
    """3D"""

    @property
    def num_dims(self) -> int:
        return 3

    @property
    def dims_names(self) -> List[str]:
        return ["x", "y", "z"]
