"""Feature for rotations"""

from typing import List

import torch

from prescyent.dataset.features.feature import Feature


class Rotation(Feature):
    """base class used for conversion"""

    def __init__(self, ids: List | range, distance_unit="rad") -> None:
        super().__init__(ids, distance_unit)

    @property
    def name(self) -> str:
        return "Rotation"

    def _is_convertible(self, __value: object) -> bool:
        return isinstance(__value, Rotation)


class RotationEuler(Rotation):
    """euler roll pitch yaw representation"""

    @property
    def num_dims(self) -> int:
        return 3

    @property
    def dims_names(self) -> List[str]:
        return ["roll", "pitch", "yaw"]


class RotationQuat(Rotation):
    """quaternion x, y, z, w representation"""

    @property
    def must_post_process(self) -> bool:
        return True

    def post_process(self, quaternion_t: torch.Tensor) -> torch.Tensor:
        """normalise a quaternion as postprocessing"""
        return quaternion_t / quaternion_t.norm(dim=-1, keepdim=True)

    @property
    def num_dims(self) -> int:
        return 4

    @property
    def dims_names(self) -> List[str]:
        return ["qx", "qy", "qz", "qw"]

    def get_distance(
        self, tensor_a: torch.Tensor, tensor_b: torch.Tensor
    ) -> torch.Tensor:
        inner_product = torch.sum(tensor_a * tensor_b, dim=-1)
        inner_product = 2 * torch.square(inner_product) - 1
        # clamp before arcos
        inner_product = torch.clamp(inner_product, min=-1, max=1)
        radian_distance = torch.arccos(inner_product)
        return radian_distance


class RotationRep6D(Rotation):
    """Continuous minimal representation from the rotmatrix, from:
    Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H. (2020).
    On the continuity of rotation representations in neural networks.
    arXiv preprint arXiv:1812.07035."""

    @property
    def num_dims(self) -> int:
        return 6

    @property
    def dims_names(self) -> List[str]:
        return ["x1", "x2", "x3", "y1", "y2", "y3"]


def is_orthonormal_matrix(R, epsilon=1e-7):
    """
    Test if matrices are orthonormal.

    Args:
        R (...xDxD tensor): batch of square matrices.
        epsilon: tolerance threshold.
    Returns:
        boolean tensor (shape ...).

    """
    R = R.clone().reshape(-1, 3, 3)
    B, D, D1 = R.shape
    assert D == D1, "Input should be a BxDxD batch of matrices."
    errors = torch.norm(
        R @ R.transpose(-1, -2) - torch.eye(D, device=R.device, dtype=R.dtype),
        dim=[-2, -1],
    )
    return torch.all(errors < epsilon)


def is_rotation_matrix(R, epsilon=1e-7):
    """
    Test if matrices are rotation matrices.

    Args:
        R (...xDxD tensor): batch of square matrices.
        epsilon: tolerance threshold.
    Returns:
        boolean tensor (shape ...).
    """
    if not is_orthonormal_matrix(R, epsilon):
        return False
    return torch.all(torch.det(R) > 0)


class RotationRotMat(Rotation):
    """rotation matrix"""

    @property
    def num_dims(self) -> int:
        return 9

    @property
    def dims_names(self) -> List[str]:
        return ["x1", "x2", "x3", "y1", "y2", "y3", "z1", "z2", "z3"]

    def get_distance(
        self, tensor_a: torch.Tensor, tensor_b: torch.Tensor
    ) -> torch.Tensor:
        # assert rotmatrices are valid
        rotmatrix_a = torch.reshape(tensor_a, [*tensor_a.shape[:-1], 3, 3])
        rotmatrix_b = torch.reshape(tensor_b, [*tensor_b.shape[:-1], 3, 3])
        # test_a = is_rotation_matrix(rotmatrix_a)
        # test_b = is_rotation_matrix(rotmatrix_b)
        # Get the transition matrix between A and B
        R_diffs = rotmatrix_a @ rotmatrix_b.transpose(-1, -2)
        # Get the angle of the rotation from the matrix
        all_traces = R_diffs.diagonal(dim1=-2, dim2=-1).sum(-1)
        all_traces = (all_traces - 1) / 2
        # Clip the trace to ensure it is within the valid range for arcos
        all_traces = torch.clamp(all_traces, -1.0, 1.0)
        # Compute the loss
        radian_distance = torch.arccos(all_traces)
        return radian_distance
