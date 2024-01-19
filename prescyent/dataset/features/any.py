from typing import List

import torch

from prescyent.dataset.features.feature import Feature


class Any(Feature):

    @property
    def num_dims(self) -> int:
        return -1

    @property
    def dims_names(self) -> List[str]:
        return [f"feature_{i}" for i in range(self.ids)]

    def _is_convertible(self, __value: object) -> bool:
        return isinstance(__value, Any) and len(self.ids) > len(__value.ids)
