"""Base class for features, checking class constraints and tensor's ids"""
from abc import abstractmethod
from typing import List, Union

import torch


class Feature(dict):
    """
    Base class with equivalence methods and checks on constructor.
    We inherit from dict for serialization
    """

    ids: List[int]
    distance_unit: str

    def __init__(self, ids: Union[List, range], distance_unit=None) -> None:
        if isinstance(ids, range):
            ids = list(ids)
        if len(ids) > len(set(ids)):
            raise AttributeError(f"All members of ids {ids} must be unique")
        self.ids = ids
        if self.num_dims >= 0 and len(ids) != self.num_dims:
            raise AttributeError(
                "lenght of provided ids mismatch expect size "
                f"{self.num_dims} for feature {self.__class__.__name__}"
            )
        self.distance_unit = distance_unit
        dict.__init__(self, ids=ids, name=self.__class__.__name__)

    def __eq__(self, __value: object) -> bool:
        return (
            self.__class__.__name__ == __value.__class__.__name__
            and self.ids == __value.ids
        )

    def __gt__(self, other):
        return (
            f"{self.__class__.__name__}_{self.ids}"
            > f"{other.__class__.__name__}_{other.ids}"
        )

    def _is_alike(self, __value: object) -> bool:
        return self.__class__.__name__ == __value.__class__.__name__ and len(
            self.ids
        ) == len(__value.ids)

    @property
    def must_post_process(self) -> bool:
        return False

    @property
    @abstractmethod
    def num_dims(self) -> int:
        return NotImplemented

    @property
    @abstractmethod
    def dims_names(self) -> List[str]:
        return NotImplemented

    @abstractmethod
    def get_distance(self, tensor_a: torch.Tensor, tensor_b: torch.Tensor):
        return NotImplemented
