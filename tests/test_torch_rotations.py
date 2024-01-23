import copy
import unittest

import torch

from prescyent.dataset import Trajectory
from prescyent.dataset.features import (
    RotationEuler,
    RotationQuat,
    RotationRep6D,
    RotationRotMat,
)


class TestTrajRotations(unittest.TestCase):
    def test_conversion_sanity_check(self):
        """tests rotations representations with 3 base quaternions and 0, 0, 0 x,y,z translation"""
        positions_quaternions = [
            [0, 0, 0, 1],
            # x,y,z and qx, qy, qz, qw for identity rotation
            [0.7071, 0, 0, 0.7071],
            # rotation of 90 degrees around the y-axis.
            [0.5, 0.5, 0.5, 0.5],
            # rotation of 120 degrees around axis equally weighted
        ]
        pos_sequence = torch.FloatTensor([positions_quaternions])
        traj = Trajectory(pos_sequence, 1, [RotationQuat(range(4))])
        quat_tensor = copy.deepcopy(traj.tensor)
        traj.convert_tensor_features([RotationRotMat(range(9))])
        traj.convert_tensor_features([RotationQuat(range(4))])
        self.assertTrue(torch.allclose(traj.tensor, quat_tensor))
        traj.convert_tensor_features([RotationRep6D(range(6))])
        traj.convert_tensor_features([RotationQuat(range(4))])
        self.assertTrue(torch.allclose(traj.tensor, quat_tensor))
        traj.convert_tensor_features([RotationEuler(range(3))])
        traj.convert_tensor_features([RotationQuat(range(4))])
        self.assertTrue(torch.allclose(traj.tensor, quat_tensor))
