"""add 6d rotation representation from
    Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H. (2020).
    On the continuity of rotation representations in neural networks.
    arXiv preprint arXiv:1812.07035.
to scipy.spatial.transform import Rotation"""

import numpy.typing as npt
import numpy as np
from scipy.spatial.transform import Rotation


def normalize_with_numpy(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


class Rotation6d(Rotation):
    @classmethod
    def from_rep6d(cls, rep6d: npt.ArrayLike) -> Rotation:
        a1 = rep6d[:, 0]
        a2 = rep6d[:, 1]
        b1 = normalize_with_numpy(a1)
        b2 = normalize_with_numpy(a2 - np.dot(b1, a2) * b1)
        b3 = np.cross(b1, b2)
        matrix = np.concatenate(
            (b1[:, np.newaxis], b2[:, np.newaxis], b3[:, np.newaxis]), 1
        ).reshape(3, 3)
        return cls.from_matrix(matrix)

    def as_rep6d(self) -> np.ndarray:
        rotmatrix = self.as_matrix()
        rep6d = np.delete(rotmatrix, -1, axis=-1)  # remove last dimension on last axis
        return rep6d

    def __eq__(self, __value: object) -> bool:
        if isinstance(__value, Rotation):
            return np.allclose(self.as_matrix(), __value.as_matrix())
        return False
