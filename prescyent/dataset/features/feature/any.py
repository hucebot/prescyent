"""Default feature without constraints and convertions"""
from typing import List

import torch

from prescyent.dataset.features.feature import Feature


class Any(Feature):
    @property
    def name(self) -> str:
        return "Any"

    @property
    def num_dims(self) -> int:
        return -1

    @property
    def dims_names(self) -> List[str]:
        return [f"feature_{i}" for i in range(self.ids)]

    def _is_convertible(self, __value: object) -> bool:
        return isinstance(__value, Any) and len(self.ids) > len(__value.ids)

    def get_distance(
        self, tensor_a: torch.Tensor, tensor_b: torch.Tensor
    ) -> torch.Tensor:
        # Chose to treat Any as a coordinate for now
        return torch.sqrt(torch.sum(torch.square(tensor_b - tensor_a), dim=-1))
