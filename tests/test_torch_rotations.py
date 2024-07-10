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
from prescyent.dataset.features.rotation_methods import convert_to_rotmatrix


class TestTrajRotations(unittest.TestCase):
    def test_conversion_sanity_check(self):
        """tests rotations representations with 3 base quaternions and 0, 0, 0 x,y,z translation"""
        positions_quaternions = [
            [0, 0, 0, 1],
            # # x,y,z and qx, qy, qz, qw for identity rotation
            [0.7071, 0, 0, 0.7071],
            # # rotation of 90 degrees around the y-axis.
            [0.5, 0.5, 0.5, 0.5],
            # rotation of 120 degrees around axis equally weighted
            [0, 0.9999995, 0, 0.0010101],
            # rotation of 179.88 degrees around x axis
        ]
        pos_sequence = torch.FloatTensor([positions_quaternions])
        traj = Trajectory(pos_sequence, 1, [RotationQuat(range(4))])
        quat_tensor = copy.deepcopy(traj.tensor)
        traj.convert_tensor_features([RotationRotMat(range(9))])
        traj.convert_tensor_features([RotationQuat(range(4))])
        self.assertTrue(torch.allclose(traj.tensor, quat_tensor, atol=1e-6))
        traj.convert_tensor_features([RotationRep6D(range(6))])
        traj.convert_tensor_features([RotationQuat(range(4))])
        self.assertTrue(torch.allclose(traj.tensor, quat_tensor, atol=1e-6))
        traj.convert_tensor_features([RotationEuler(range(3))])
        traj.convert_tensor_features([RotationQuat(range(4))])
        self.assertTrue(torch.allclose(traj.tensor, quat_tensor, atol=1e-6))


class TestPostProcesses(unittest.TestCase):
    def test_conversion_sanity_check(self):
        positions_quaternions = [
            [0, 0, 0, 1],
            # # x,y,z and qx, qy, qz, qw for identity rotation
            [0.7071, 0, 0, 0.7071],
            # # rotation of 90 degrees around the y-axis.
            [0.5, 0.5, 0.5, 0.5],
            # rotation of 120 degrees around axis equally weighted
            [0, 0.9999995, 0, 0.0010101],
            # rotation of 179.88 degrees around x axis
        ]
        pos_sequence = torch.FloatTensor([positions_quaternions])
        feat = RotationQuat(range(4))
        norm_pos = feat.post_process(pos_sequence)
        self.assertTrue(torch.allclose(norm_pos, pos_sequence))
        pos_sequence = convert_to_rotmatrix(
            pos_sequence
        )  # postprocess is included in higher level methods
        feat = RotationRotMat(range(9))
        norm_pos = feat.post_process(pos_sequence)
        self.assertTrue(torch.allclose(norm_pos, pos_sequence))
