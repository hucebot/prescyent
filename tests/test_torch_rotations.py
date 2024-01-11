import copy
import unittest

import torch

from prescyent.dataset.trajectories.position_trajectory import PositionsTrajectory
from prescyent.dataset.trajectories.features.position import Position
from prescyent.utils.enums.rotation_representation import RotationRepresentation


class TestPositionTrajRotations(unittest.TestCase):
    def test_conversion_sanity_check(self):
        """tests rotations representations with 3 base quaternions and 0, 0, 0 x,y,z translation"""
        positions_quaternions = [
            [0, 0, 0, 0, 0, 0, 1],
            # x,y,z and qx, qy, qz, qw for identity rotation
            [0, 0, 0, 0.7071, 0, 0, 0.7071],
            # rotation of 90 degrees around the y-axis.
            [0, 0, 0, 0.5, 0.5, 0.5, 0.5],
            # rotation of 120 degrees around axis equally weighted
        ]
        pos_sequence = torch.FloatTensor([positions_quaternions])
        traj = PositionsTrajectory(
            pos_sequence, RotationRepresentation.QUATERNIONS, frequency=1
        )
        quat_tensor = copy.deepcopy(traj.tensor)
        traj.rotation_representation = RotationRepresentation.ROTMATRICES
        traj.rotation_representation = RotationRepresentation.QUATERNIONS
        self.assertTrue(torch.allclose(traj.tensor, quat_tensor))
        traj.rotation_representation = RotationRepresentation.REP6D
        traj.rotation_representation = RotationRepresentation.QUATERNIONS
        self.assertTrue(torch.allclose(traj.tensor, quat_tensor))
        traj.rotation_representation = RotationRepresentation.EULER
        traj.rotation_representation = RotationRepresentation.QUATERNIONS
        self.assertTrue(torch.allclose(traj.tensor, quat_tensor))
