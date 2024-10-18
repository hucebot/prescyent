import unittest
import torch

import prescyent.dataset.features as tensor_features
from prescyent.utils.tensor_manipulation import (
    cat_list_with_seq_idx,
    trajectory_tensor_get_dim_limits,
)


class TestFlattenListPreds(unittest.TestCase):
    def test_bad_shapes(self):
        input_pred = torch.arange(3 * 3).view(3, 3)
        with self.assertRaises(AttributeError):
            _ = cat_list_with_seq_idx(input_pred)

    def test_flatten(self):
        input_pred = torch.arange(3 * 3 * 3 * 3).view(3, 3, 3, 3)
        expected_output = torch.Tensor(
            [
                [[18.0, 19.0, 20.0], [21.0, 22.0, 23.0], [24.0, 25.0, 26.0]],
                [[45.0, 46.0, 47.0], [48.0, 49.0, 50.0], [51.0, 52.0, 53.0]],
                [[72.0, 73.0, 74.0], [75.0, 76.0, 77.0], [78.0, 79.0, 80.0]],
            ]
        )
        output_pred = cat_list_with_seq_idx(input_pred)
        self.assertTrue(torch.equal(expected_output, output_pred))


class TestMinMaxDim(unittest.TestCase):
    def test_logic(self):
        a = [
            [
                [1, -2, 3],
                [2, -1, 3],
            ],
            [
                [1, 0, 3],
                [2, -2, 5],
            ],
            [
                [2, -2, 3],
                [1, -2, 3],
            ],
            [
                [3, -2, 3],
                [4, -2, -3],
            ],
            [
                [-1, -2, 3],
                [1, -5, 3],
            ],
        ]
        expected_max = torch.FloatTensor([4, 0, 5])
        expected_min = torch.FloatTensor([-1, -5, -3])
        min_t, max_t = trajectory_tensor_get_dim_limits(torch.FloatTensor(a))
        self.assertTrue(torch.equal(expected_min, min_t))
        self.assertTrue(torch.equal(expected_max, max_t))


class TestFeatureConversion(unittest.TestCase):
    def test_swap(self):
        tensor = torch.Tensor([[[0, 1, 2]]])
        truth = torch.Tensor([[[0, 2, 1]]])
        test = tensor_features.convert_tensor_features_to(
            tensor, [tensor_features.Any(range(3))], [tensor_features.Any([0, 2, 1])]
        )
        self.assertTrue(torch.equal(test, truth))
        truth = torch.Tensor([[[0, 1]]])
        test = tensor_features.convert_tensor_features_to(
            tensor, [tensor_features.Any(range(3))], [tensor_features.Any(range(2))]
        )
        self.assertTrue(torch.equal(test, truth))
        with self.assertRaises(AttributeError) as context:
            test = tensor_features.convert_tensor_features_to(
                tensor,
                tensor_features.Features([tensor_features.Any(range(3))]),
                tensor_features.Features([tensor_features.Any(range(4))]),
            )
        self.assertTrue("Cannot convert feature" in str(context.exception))

    def test_rotations(self):
        tensor = torch.Tensor([[[0, 0, 0, 0.7071, 0, 0, 0.7071]]])
        test_rotmat = tensor_features.convert_tensor_features_to(
            tensor,
            [
                tensor_features.CoordinateXYZ(range(3)),
                tensor_features.RotationQuat(range(3, 7)),
            ],
            [
                tensor_features.CoordinateXYZ(range(3)),
                tensor_features.RotationRotMat(range(3, 12)),
            ],
        )
        test_rep6d = tensor_features.convert_tensor_features_to(
            test_rotmat,
            [
                tensor_features.CoordinateXYZ(range(3)),
                tensor_features.RotationRotMat(range(3, 12)),
            ],
            [
                tensor_features.CoordinateXYZ(range(3)),
                tensor_features.RotationRep6D(range(3, 9)),
            ],
        )
        test_euler = tensor_features.convert_tensor_features_to(
            test_rep6d,
            [
                tensor_features.CoordinateXYZ(range(3)),
                tensor_features.RotationRep6D(range(3, 9)),
            ],
            [
                tensor_features.CoordinateXYZ(range(3)),
                tensor_features.RotationEuler(range(3, 6)),
            ],
        )
        test_quat = tensor_features.convert_tensor_features_to(
            test_euler,
            [
                tensor_features.CoordinateXYZ(range(3)),
                tensor_features.RotationEuler(range(3, 6)),
            ],
            [
                tensor_features.CoordinateXYZ(range(3)),
                tensor_features.RotationQuat(range(3, 7)),
            ],
        )
        self.assertTrue(torch.allclose(tensor, test_quat))
