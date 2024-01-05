"""add 6d rotation representation from
    Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H. (2020).
    On the continuity of rotation representations in neural networks.
    arXiv preprint arXiv:1812.07035.
to scipy.spatial.transform import Rotation"""

import numpy.typing as npt
import numpy as np
from scipy.spatial.transform import Rotation
import torch


def normalize_with_torch(v):
    norm = torch.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def rep6d_to_rotmatrix(rep6d: torch.Tensor) -> torch.Tensor:
    if not isinstance(rep6d, torch.Tensor):
        rep6d = torch.from_numpy(rep6d)
    if len(rep6d.shape) == 1:
        rep6d = rep6d.reshape(3, 2)
    a1 = rep6d[:, 0]
    a2 = rep6d[:, 1]
    b1 = normalize_with_torch(a1)
    b2 = normalize_with_torch(a2 - torch.dot(b1, a2) * b1)
    b3 = torch.cross(b1, b2)
    matrix = torch.cat(
        (b1.unsqueeze(1), b2.unsqueeze(1), b3.unsqueeze(1)), dim=1
    ).reshape(3, 3)
    return matrix


def rotmatrix_to_rep6d(rotmatrix: torch.Tensor) -> torch.Tensor:
    if not isinstance(rotmatrix, torch.Tensor):
        rotmatrix = torch.from_numpy(rotmatrix)
    if len(rotmatrix.shape) == 1:
        rotmatrix = rotmatrix.reshape(3, 3)
    rep6d = torch.narrow(
        rotmatrix, 1, 0, rotmatrix.shape[1] - 1
    )  # remove last dimension on last axis
    return rep6d


class Rotation6d(Rotation):
    @classmethod
    def from_rep6d(cls, rep6d: npt.ArrayLike) -> Rotation:
        matrix = rep6d_to_rotmatrix(rep6d=rep6d)
        return cls.from_matrix(matrix)

    def as_rep6d(self) -> np.ndarray:
        rep6d = rotmatrix_to_rep6d(self.as_matrix()).numpy()
        return rep6d

    def __eq__(self, __value: object) -> bool:
        if isinstance(__value, Rotation):
            return np.allclose(self.as_matrix(), __value.as_matrix())
        return False
