"""Feature used to represent coordinates from 1D to 3D"""
from typing import List, Union

import torch

from prescyent.dataset.features.feature import Feature


class Coordinate(Feature):
    """parent class for coordinates, used for conversion"""

    def __init__(
        self,
        ids: Union[List, range],
        distance_unit: str = "m",
        name: str = "Coordinate",
    ) -> None:
        self.name = name
        super().__init__(ids, distance_unit)

    def _is_convertible(self, __value: object) -> bool:
        return isinstance(__value, Coordinate) and len(self.ids) > len(__value.ids)

    def get_distance(
        self, tensor_a: torch.Tensor, tensor_b: torch.Tensor
    ) -> torch.Tensor:
        return torch.norm(tensor_b - tensor_a, dim=-1)
        # using frobenius norm is equivalent to =>
        # torch.sqrt(torch.sum(torch.square(tensor_b - tensor_a), dim=-1))


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
