"""Default feature without constraints and convertions"""
from typing import List, Union

import torch

from prescyent.dataset.features.feature import Feature


class Any(Feature):
    def __init__(self, ids: Union[List, range], distance_unit="_", name="Any") -> None:
        self.name = name
        super().__init__(ids, distance_unit)

    @property
    def num_dims(self) -> int:
        return -1

    @property
    def dims_names(self) -> List[str]:
        return [f"feature_{i}" for i in range(len(self.ids))]

    def _is_convertible(self, __value: object) -> bool:
        return isinstance(__value, Any) and len(self.ids) > len(__value.ids)

    def get_distance(
        self, tensor_a: torch.Tensor, tensor_b: torch.Tensor
    ) -> torch.Tensor:
        # Chose to treat Any as a coordinate for now
        return torch.sqrt(torch.sum(torch.square(tensor_b - tensor_a), dim=-1))
