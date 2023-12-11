from typing import List, Union, Tuple

import torch
import numpy as np
from scipy.spatial.transform import Rotation

from prescyent.dataset.trajectories.features.coordinates import Coordinates
from prescyent.utils.enums import RotationRepresentation
from prescyent.utils.quaternion_manipulation import get_distance_rotations


DEFAULT_EULER_SEQ = "ZYX"
DEFAULT_EULER_IS_DEGREES = False


class Position:
    coordinates: Coordinates
    rotation: Rotation
    rotation_representation: RotationRepresentation

    def __init__(
        self,
        x: float,
        y: float = None,
        z: float = None,
        rotation: Rotation = None,
        rotation_representation: RotationRepresentation = None,
    ) -> None:
        self.coordinates = Coordinates(x, y, z)
        self.rotation = rotation
        self.rotation_representation = rotation_representation

    @classmethod
    def init_from_quaternion(
        cls,
        x: float,
        y: float = None,
        z: float = None,
        qx: float = None,
        qy: float = None,
        qz: float = None,
        qw: float = None,
    ) -> None:
        rotation = Rotation.from_quat([qx, qy, qz, qw])
        return cls(x, y, z, rotation, RotationRepresentation.QUATERNIONS)

    @classmethod
    def init_from_euler(
        cls,
        x: float,
        y: float = None,
        z: float = None,
        e1: float = None,
        e2: float = None,
        e3: float = None,
        euler_seq: str = DEFAULT_EULER_SEQ,
        degrees: bool = DEFAULT_EULER_IS_DEGREES,
    ) -> None:
        rotation = Rotation.from_euler(euler_seq, [e1, e2, e3], degrees=degrees)
        return cls(x, y, z, rotation, RotationRepresentation.EULER)

    def num_dims(self) -> int:
        rotation_dim = 0
        if self.rotation is not None:
            rotation_dim = len(self._get_rotation_array())
        return self.coordinates.num_dims + rotation_dim

    def get_tensor(self) -> torch.Tensor:
        tensor = self.coordinates.get_tensor()
        if self.rotation is not None:
            rotation_array = self._get_rotation_array()
            tensor = torch.cat(
                (
                    tensor,
                    torch.from_numpy(rotation_array),
                ),
                dim=0,
            )
        return tensor

    def _get_rotation_array(self) -> np.ndarray:
        if self.rotation is not None:
            if self.rotation_representation == RotationRepresentation.EULER:
                rotation_array = self.rotation.as_euler(
                    DEFAULT_EULER_SEQ, degrees=DEFAULT_EULER_IS_DEGREES
                )
            elif self.rotation_representation == RotationRepresentation.ROTMATRICES:
                rotation_array = self.rotation.as_matrix()
                rotation_array = np.transpose(rotation_array)
                rotation_array = rotation_array.flatten()
            elif self.rotation_representation == RotationRepresentation.QUATERNIONS:
                rotation_array = self.rotation.as_quat()
            elif self.rotation_representation == RotationRepresentation.ROTVECTORS:
                rotation_array = self.rotation.as_rotvec()
            elif self.rotation_representation == RotationRepresentation.RODRIGUES:
                rotation_array = self.rotation.as_mrp()
            else:
                raise NotImplementedError(
                    f"No tensor representation available for {self.rotation_representation}"
                )
            return rotation_array
        return None

    def dim_names(self) -> List[str]:
        dim_names = self.coordinates.dim_names()
        if self.rotation is not None:
            if self.rotation_representation == RotationRepresentation.QUATERNIONS:
                dim_names += ["qx", "qy", "qz", "qw"]
            elif self.rotation_representation == RotationRepresentation.EULER:
                dim_names += [f"e{dim}" for dim in DEFAULT_EULER_SEQ]
            elif self.rotation_representation == RotationRepresentation.ROTMATRICES:
                dim_names += ["x1", "x2", "x3", "y1", "y2", "y3", "z1", "z2", "z3"]
        return dim_names

    @classmethod
    def get_from_tensor(
        cls,
        tensor: Union[torch.Tensor, List[float]],
        rotation_representation: RotationRepresentation,
    ):
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.tolist()
        if rotation_representation == RotationRepresentation.QUATERNIONS:
            return cls.init_from_quaternion(*list(tensor))
        if len(tensor) <= 3:
            return Position(*list(tensor))
        if rotation_representation == RotationRepresentation.EULER:
            return cls.init_from_euler(*list(tensor))
        raise NotImplementedError()

    def calc_distance(self, other: object) -> Tuple[float, float]:
        assert isinstance(other, Position)
        coordinate_distance = self.coordinates.calc_distance(other.coordinates)
        rotation_distance = 0
        if self.rotation is not None and other.rotation is not None:
            rotation_distance = get_distance_rotations(self.rotation, other.rotation)
        return coordinate_distance, rotation_distance
