import copy
import unittest

import torch

from prescyent.dataset.trajectories.position_trajectory import PositionsTrajectory
from prescyent.dataset.trajectories.features.position import Position
from prescyent.utils.enums.rotation_representation import RotationRepresentation


class TestPositionTrajRotations(unittest.TestCase):
    def test_conversion_sanity_check(self):
        """tests rotations representations with 3 base quaternions and 0, 0, 0 x,y,z translation"""
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
        sequence = [positions_quaternions]
        traj = PositionsTrajectory(sequence_of_positions=sequence, frequency=1)
        quat_tensor = copy.deepcopy(traj.tensor)
        for test_rotation in test_rotations:
            traj.rotation_representation = test_rotation
            test_tensor = traj.tensor
            test_frames = []
            for frame in test_tensor:
                test_positions = []
                for point in frame:
                    test_positions.append(
                        Position.get_from_tensor(point, test_rotation)
                    )
                test_frames.append(test_positions)
            test_traj = PositionsTrajectory(
                sequence_of_positions=test_frames, frequency=1
            )
            test_traj.rotation_representation = RotationRepresentation.QUATERNIONS
            self.assertTrue(torch.allclose(test_traj.tensor, quat_tensor))
