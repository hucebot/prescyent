from abc import abstractmethod
from typing import List, Any, Union


class Feature(dict):
    ids: List[int]

    def __init__(self, ids:Union[List, range]) -> None:
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
        dict.__init__(self, ids=ids, name=self.__class__.__name__)

    def __eq__(self, __value: object) -> bool:
        return self.__class__.__name__ == __value.__class__.__name__ and self.ids == __value.ids

    def _is_alike(self, __value: object) -> bool:
        return self.__class__.__name__ == __value.__class__.__name__ and len(self.ids) == len(__value.ids)

    @property
    @abstractmethod
    def num_dims(self) -> int:
        return NotImplemented

    @property
    @abstractmethod
    def dims_names(self) -> List[str]:
        return NotImplemented
