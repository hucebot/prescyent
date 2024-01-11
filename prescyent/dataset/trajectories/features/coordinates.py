from typing import List

import torch


class Coordinates:
    x: float
    y: float
    z: float

    def __init__(self, x: float, y: float = None, z: float = None) -> None:
        self.x = x
        self.y = y
        self.z = z

    def num_dims(self) -> int:
        return len([i for i in [self.x, self.y, self.z] if i is not None])

    def get_tensor(self) -> torch.Tensor:
        return torch.FloatTensor([i for i in [self.x, self.y, self.z] if i is not None])

    def dim_names(self) -> List[str]:
        dim_names = ["x"]
        if self.y is not None:
            dim_names.append("y")
        if self.z is not None:
            dim_names.append("z")
        return dim_names

    def calc_distance(self, other: object) -> float:
        assert isinstance(other, Coordinates)
        distances = []
        if self.x and other.x:
            distances.append(pow(self.x - other.x, 2))
        if self.y and other.y:
            distances.append(pow(self.y - other.y, 2))
        if self.z and other.z:
            distances.append(pow(self.z - other.z, 2))
        return pow(sum(distances), 0.5)