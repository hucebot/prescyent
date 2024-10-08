import copy
import unittest

import torch

from prescyent.dataset import Trajectory
from prescyent.dataset.features import (
    Features,
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
            # x,y,z and qx, qy, qz, qw for identity rotation
            [0.7071, 0, 0, 0.7071],
            # rotation of 90 degrees around the y-axis.
            [0.5, 0.5, 0.5, 0.5],
            # rotation of 120 degrees around axis equally weighted
            [0, 0.9999995, 0, 0.0010101],
            # rotation of 179.88 degrees around x axis
        ]
        pos_sequence = torch.FloatTensor([positions_quaternions])
        traj = Trajectory(pos_sequence, 1, Features([RotationQuat(range(4))]))
        quat_tensor = copy.deepcopy(traj.tensor)
        traj.convert_tensor_features(Features([RotationRotMat(range(9))]))
        traj.convert_tensor_features(Features([RotationQuat(range(4))]))
        self.assertTrue(torch.allclose(traj.tensor, quat_tensor, atol=1e-6))
        traj.convert_tensor_features(Features([RotationRep6D(range(6))]))
        traj.convert_tensor_features(Features([RotationQuat(range(4))]))
        self.assertTrue(torch.allclose(traj.tensor, quat_tensor, atol=1e-6))
        traj.convert_tensor_features(Features([RotationEuler(range(3))]))
        traj.convert_tensor_features(Features([RotationQuat(range(4))]))
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
        self.assertTrue(torch.allclose(norm_pos, pos_sequence, atol=1e-4))
        pos_sequence = convert_to_rotmatrix(
            pos_sequence
        )  # postprocess is included in higher level methods
        feat = RotationRotMat(range(9))
        norm_pos = feat.post_process(pos_sequence)
        self.assertTrue(torch.allclose(norm_pos, pos_sequence, atol=1e-4))


class TestDistanceCalculation(unittest.TestCase):
    def test_get_distance_quat(self):
        qi = torch.FloatTensor([0, 0, 0, 1])
        qri = torch.FloatTensor([0, 0, 0, -1])
        # qx, qy, qz, qw for identity rotation
        q1 = torch.FloatTensor([0.7071, 0, 0, 0.7071])
        qr1 = torch.FloatTensor([-0.7071, 0, 0, -0.7071])
        # rotation of 90 degrees around the x-axis.
        q2 = torch.FloatTensor([0, 0.7071, 0, 0.7071])
        qr2 = torch.FloatTensor([0, -0.7071, 0, -0.7071])
        # rotation of 90 degrees around the y-axis.
        q3 = torch.FloatTensor([0.5, 0.5, 0.5, 0.5])
        qr3 = torch.FloatTensor([-0.5, -0.5, -0.5, -0.5])
        # rotation of 120 degrees around all axis equally weighted
        q4 = torch.FloatTensor([1, 0, 0, 0])
        qr4 = torch.FloatTensor([-1, 0, 0, 0])
        feat = RotationQuat(range(4))
        dist = feat.get_distance(qi, qri)
        self.assertTrue(
            torch.allclose(dist, torch.FloatTensor([0.0005]).squeeze(), atol=1e-4)
        )
        dist = feat.get_distance(qi, q1)
        self.assertTrue(
            torch.allclose(dist, torch.FloatTensor([1.57081]).squeeze(), atol=1e-4)
        )
        dist = feat.get_distance(qi, qr1)
        self.assertTrue(
            torch.allclose(dist, torch.FloatTensor([1.57081]).squeeze(), atol=1e-4)
        )
        dist = feat.get_distance(qi, q2)
        self.assertTrue(
            torch.allclose(dist, torch.FloatTensor([1.57081]).squeeze(), atol=1e-4)
        )
        dist = feat.get_distance(qi, qr2)
        self.assertTrue(
            torch.allclose(dist, torch.FloatTensor([1.57081]).squeeze(), atol=1e-4)
        )
        dist = feat.get_distance(qi, q3)
        self.assertTrue(
            torch.allclose(dist, torch.FloatTensor([2.094395]).squeeze(), atol=1e-4)
        )
        dist = feat.get_distance(qi, qr3)
        self.assertTrue(
            torch.allclose(dist, torch.FloatTensor([2.094395]).squeeze(), atol=1e-4)
        )
        dist = feat.get_distance(qi, q4)
        self.assertTrue(
            torch.allclose(dist, torch.FloatTensor([3.141104]).squeeze(), atol=1e-4)
        )
        dist = feat.get_distance(qi, qr4)
        self.assertTrue(
            torch.allclose(dist, torch.FloatTensor([3.141104]).squeeze(), atol=1e-4)
        )

    def test_get_distance_rotmat(self):
        qi = torch.FloatTensor([[0, 0, 0, 1]])
        qri = torch.FloatTensor([[0, 0, 0, -1]])
        # qx, qy, qz, qw for identity rotation
        q1 = torch.FloatTensor([[0.7071, 0, 0, 0.7071]])
        qr1 = torch.FloatTensor([[-0.7071, 0, 0, -0.7071]])
        # rotation of 90 degrees around the x-axis.
        q2 = torch.FloatTensor([[0, 0.7071, 0, 0.7071]])
        qr2 = torch.FloatTensor([[0, -0.7071, 0, -0.7071]])
        # rotation of 90 degrees around the y-axis.
        q3 = torch.FloatTensor([[0.5, 0.5, 0.5, 0.5]])
        qr3 = torch.FloatTensor([[-0.5, -0.5, -0.5, -0.5]])
        # rotation of 120 degrees around all axis equally weighted
        q4 = torch.FloatTensor([[1, 0, 0, 0]])
        qr4 = torch.FloatTensor([[-1, 0, 0, 0]])
        qi = convert_to_rotmatrix(qi)
        qri = convert_to_rotmatrix(qri)
        q1 = convert_to_rotmatrix(q1)
        qr1 = convert_to_rotmatrix(qr1)
        q2 = convert_to_rotmatrix(q2)
        qr2 = convert_to_rotmatrix(qr2)
        q3 = convert_to_rotmatrix(q3)
        qr3 = convert_to_rotmatrix(qr3)
        q4 = convert_to_rotmatrix(q4)
        qr4 = convert_to_rotmatrix(qr4)
        feat = RotationRotMat(range(9))
        dist = feat.get_distance(qi, qri)
        self.assertTrue(
            torch.allclose(dist, torch.FloatTensor([0.0005]).squeeze(), atol=1e-4)
        )
        dist = feat.get_distance(qi, q1)
        self.assertTrue(
            torch.allclose(dist, torch.FloatTensor([1.57081]).squeeze(), atol=1e-4)
        )
        dist = feat.get_distance(qi, qr1)
        self.assertTrue(
            torch.allclose(dist, torch.FloatTensor([1.57081]).squeeze(), atol=1e-4)
        )
        dist = feat.get_distance(qi, q2)
        self.assertTrue(
            torch.allclose(dist, torch.FloatTensor([1.57081]).squeeze(), atol=1e-4)
        )
        dist = feat.get_distance(qi, qr2)
        self.assertTrue(
            torch.allclose(dist, torch.FloatTensor([1.57081]).squeeze(), atol=1e-4)
        )
        dist = feat.get_distance(qi, q3)
        self.assertTrue(
            torch.allclose(dist, torch.FloatTensor([2.094395]).squeeze(), atol=1e-4)
        )
        dist = feat.get_distance(qi, qr3)
        self.assertTrue(
            torch.allclose(dist, torch.FloatTensor([2.094395]).squeeze(), atol=1e-4)
        )
        dist = feat.get_distance(qi, q4)
        self.assertTrue(
            torch.allclose(dist, torch.FloatTensor([3.141104]).squeeze(), atol=1e-4)
        )
        dist = feat.get_distance(qi, qr4)
        self.assertTrue(
            torch.allclose(dist, torch.FloatTensor([3.141104]).squeeze(), atol=1e-4)
        )
