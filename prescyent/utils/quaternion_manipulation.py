"""Util functions for quaternions"""
import functools
import math

import numpy as np
from scipy.spatial.transform import Rotation
import torch


def scipy_rotation_conversion(function):
    """decorator to allow seamless use of some high level methods with scipy rotations"""

    @functools.wraps(function)
    def convert_scipy_to_quat(*args, **kwargs):
        q1 = args[0]
        q2 = args[1]
        if isinstance(q1, np.ndarray) and isinstance(q2, np.ndarray):
            return function(q1, q2)
        elif isinstance(q1, Rotation) and isinstance(q2, Rotation):
            q1 = q1.as_quat()
            q2 = q2.as_quat()
            res_quat = function(q1, q2)
            return Rotation.from_quat(res_quat)
        else:
            raise NotImplementedError(
                "please use consistent types for q1, q2, ",
                "Rotation or ndarray representation are supported",
            )

    return convert_scipy_to_quat


@scipy_rotation_conversion
def multiply_quaternions(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiplies two quaternions and returns the result.

    Args:
    - q1 (numpy.ndarray): A 1DArray representing the first quaternion [x, y, z, w].
    - q2 (numpy.ndarray): A 1DArray representing the second quaternion [x, y, z, w].

    Returns:
    - numpy.ndarray: The result of multiplying the two quaternions [x, y, z, w].
    """
    assert len(q1) == 4
    assert len(q2) == 4
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    return np.array([x, y, z, w])


def inverse_quaternion(q: np.ndarray) -> np.ndarray:
    """Calculate the inverse quaternion of input and returns the result.

    Args:
        q (np.ndarray): A 1DArray representing a quaternion [x, y, z, w].

    Returns:
    - numpy.ndarray: A 1DArray representing the inverse quaternions [x, y, z, w].
    """
    x, y, z, w = q
    norm_squared = x**2 + y**2 + z**2 + w**2
    return np.array(
        [-x / norm_squared, -y / norm_squared, -z / norm_squared, w / norm_squared]
    )


@scipy_rotation_conversion
def get_transformation_quaternion(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Calculate the transformation between two quaternions and returns the result.

    Args:
    - q1 (numpy.ndarray): A 1DArray representing the first quaternion [x, y, z, w].
    - q2 (numpy.ndarray): A 1DArray representing the second quaternion [x, y, z, w].

    Returns:
    - numpy.ndarray: A 1DArray representing the transformation between q1 and q2 [x, y, z, w].
    """
    q1_inv = inverse_quaternion(q1)
    transformation_quaternion = multiply_quaternions(q2, q1_inv)
    return transformation_quaternion


def get_distance_rotations(q1: Rotation, q2: Rotation, degrees=False) -> float:
    """Calculate the distance between two rotations and returns the result.

    Args:
    - q1 (Rotation): Scipy representation of the first rotation.
    - q2 (Rotation): Scipy representation of the second rotation.

    Returns:
    - float: the magnitude of the transformation quaternion between q1 and q2
    """
    transformation_quaternion = get_transformation_quaternion(q1, q2)
    magnitude = transformation_quaternion.magnitude()
    if degrees:
        magnitude = math.degrees(magnitude)
    return magnitude


def quaternion_to_rotmatrix(q: torch.Tensor) -> torch.Tensor:
    if not isinstance(q, torch.Tensor):
        q = torch.from_numpy(q)
    x, y, z, w = q[0].float(), q[1].float(), q[2].float(), q[3].float()
    return torch.FloatTensor(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
            [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
            [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
        ]
    )
