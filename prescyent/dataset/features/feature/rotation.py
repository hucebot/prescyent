"""Feature for rotations"""

from typing import List, Union

import torch

from prescyent.dataset.features.feature import Feature


class Rotation(Feature):
    """base class used for conversion"""

    def __init__(
        self,
        ids: Union[List, range],
        distance_unit: str = "rad",
        name: str = "Rotation",
    ) -> None:
        self.name = name
        super().__init__(ids, distance_unit)

    def _is_convertible(self, __value: object) -> bool:
        return isinstance(__value, Rotation)


class RotationEuler(Rotation):
    """euler roll pitch yaw representation"""

    @property
    def num_dims(self) -> int:
        """size of the feature"""
        return 3

    @property
    def dims_names(self) -> List[str]:
        """name of each dim"""
        return ["roll", "pitch", "yaw"]


class RotationQuat(Rotation):
    """quaternion x, y, z, w representation"""

    @property
    def eps(self) -> float:
        return 1e-7

    @property
    def must_post_process(self) -> bool:
        """returns true as we want to manipulate actual quaternions in the lib"""
        return True

    def post_process(self, quaternion_t: torch.Tensor) -> torch.Tensor:
        """normalise a quaternion as postprocessing

        Args:
            quaternion_t (torch.Tensor): quaternion to normalize

        Returns:
            torch.Tensor: normalized quaternion
        """
        quat_normed = quaternion_t / quaternion_t.norm(dim=-1, keepdim=True)
        # Ensure we have the quaternion with a positive w to avoid double cover
        indices = torch.nonzero(quat_normed[..., -1] < 0, as_tuple=True)
        quat_normed[indices] = -quat_normed[indices]
        return quat_normed

    @property
    def num_dims(self) -> int:
        """size of the feature"""
        return 4

    @property
    def dims_names(self) -> List[str]:
        """name of each dim"""
        return ["qx", "qy", "qz", "qw"]

    def get_distance(
        self, tensor_a: torch.Tensor, tensor_b: torch.Tensor
    ) -> torch.Tensor:
        """computes angular distance in radian

        Args:
            tensor_a (torch.Tensor): tensor to compare
            tensor_b (torch.Tensor): tensor to compare

        Returns:
            torch.Tensor: distance between the two tensors
        """
        inner_product = torch.sum(tensor_a * tensor_b, dim=-1)
        inner_product = 2 * torch.square(inner_product) - 1
        # clamp before arcos
        inner_product = torch.clamp(inner_product, min=-1 + self.eps, max=1 - self.eps)
        radian_distance = torch.arccos(inner_product)
        return radian_distance


class RotationRep6D(Rotation):
    """Continuous minimal representation from the rotmatrix, from:
    Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H. (2020).
    On the continuity of rotation representations in neural networks.
    arXiv preprint arXiv:1812.07035."""

    @property
    def num_dims(self) -> int:
        """size of the feature"""
        return 6

    @property
    def dims_names(self) -> List[str]:
        """name of each dim"""
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
    def eps(self) -> float:
        return 1e-7

    @property
    def num_dims(self) -> int:
        """size of the feature"""
        return 9

    @property
    def dims_names(self) -> List[str]:
        """name of each dim"""
        return ["x1", "x2", "x3", "y1", "y2", "y3", "z1", "z2", "z3"]

    @property
    def must_post_process(self) -> bool:
        """returns true as we want to manipulate actual rotmatrices in the lib"""
        return True

    def post_process(self, rotmat_t: torch.Tensor) -> torch.Tensor:
        """Use SVD to postprocess rotation matrices

        Args:
            rotmat_t (torch.Tensor): matrix tensor with shape like [..., 9]

        Returns:
            torch.Tensor: postprocessed rotation matrix tensor with shape like [..., 9].
        """
        rotmat_t = rotmat_t.reshape(*rotmat_t.shape[:-1], 3, 3)
        u, _, v = torch.svd(rotmat_t)
        v = torch.transpose(v, -2, -1)
        det = torch.det(torch.matmul(u, v))
        det = det.view(*rotmat_t.shape[:-2], 1, 1)
        v = torch.cat((v[..., :2, :], v[..., -1:, :] * det), -2)
        r = torch.matmul(u, v)
        return r.reshape(*rotmat_t.shape[:-2], 9)

    def get_distance(
        self, tensor_a: torch.Tensor, tensor_b: torch.Tensor
    ) -> torch.Tensor:
        """computes angular distance in radian

        Args:
            tensor_a (torch.Tensor): tensor to compare
            tensor_b (torch.Tensor): tensor to compare

        Returns:
            torch.Tensor: distance between the two tensors
        """
        # assert rotmatrices are valid
        rotmatrix_a = torch.reshape(tensor_a, [*tensor_a.shape[:-1], 3, 3])
        rotmatrix_b = torch.reshape(tensor_b, [*tensor_b.shape[:-1], 3, 3])
        # Get the transition matrix between A and B
        R_diffs = rotmatrix_a @ rotmatrix_b.transpose(-1, -2)
        # Get the angle of the rotation from the matrix
        all_traces = R_diffs.diagonal(dim1=-2, dim2=-1).sum(-1)
        all_traces = (all_traces - 1) / 2
        # Clip the trace to ensure it is within the valid range for arcos
        all_traces = torch.clamp(all_traces, -1.0 + self.eps, 1.0 - self.eps)
        # Compute the loss
        radian_distance = torch.arccos(all_traces)
        return radian_distance
