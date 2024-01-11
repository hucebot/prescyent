import copy
import unittest

import torch

from prescyent.dataset.trajectories.features.position import Position
from prescyent.utils.enums.rotation_representation import RotationRepresentation


class TestPositionRotations(unittest.TestCase):
    def test_conversion_sanity_check(self):
        """tests rotations representations with 3 basics quaternions and 0, 0, 0 x,y,z translation"""
        test_rotations = [
            RotationRepresentation.QUATERNIONS,
            RotationRepresentation.EULER,
            RotationRepresentation.ROTMATRICES,
            RotationRepresentation.REP6D,
        ]
        positions_quaternions = [
            Position.init_from_quaternion(*[0, 0, 0, 0, 0, 0, 1]),
            # x,y,z and qx, qy, qz, qw for identity rotation
            Position.init_from_quaternion(*[0, 0, 0, 0.707, 0, 0, 0.707]),
            # rotation of 90 degrees around the y-axis.
            Position.init_from_quaternion(*[0, 0, 0, 0.5, 0.5, 0.5, 0.5]),
            # rotation of 120 degrees around axis equally weighted
        ]
        for position in positions_quaternions:
            for test_rotation in test_rotations:
                tensor = position.get_tensor(test_rotation)
                test_position = Position.get_from_tensor(tensor, test_rotation)
                self.assertEqual(position.rotation, test_position.rotation)
