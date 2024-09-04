import unittest

import torch

import prescyent.dataset.features as tensor_features


class TestFeatureWiseDistance(unittest.TestCase):
    def test_return_3d_coordinate_quat_distances(self):
        a = torch.FloatTensor(
            [
                [
                    [1, 1, 1, 0, 0, 0, 1],
                    [1, 1, 1, 0, 0, 0, 1],
                    [1, 1, 1, 0, 0, 0, 1],
                    [1, 1, 1, 0, 0, 0, 1],
                    [3, 5, 5, 0, -0.4794255, 0, 0.8775826],
                ]
            ]
        )
        b = torch.FloatTensor(
            [
                [
                    [1, 1, 1, 0, 0.4794255, 0, 0.8775826],
                    [1, 1, 0, 0, 0, 0, 1],
                    [1, 1, 2, 0, 0.174941, 0.174941, 0.9689124],
                    [11, 1, 1, 0, 0.7526345, 0.3763173, 0.5403023],
                    [1, 1, 1, 0, 0.4794255, 0, 0.8775826],
                ]
            ]
        )
        truth_coord = torch.FloatTensor([[[0.0, 1.0, 1.0, 10.0, 6.0]]])
        truth_rot = torch.FloatTensor([[[1.0, 0.0, 0.5, 2.0, 2.0]]])
        feature = tensor_features.Features(
            [
                tensor_features.CoordinateXYZ(range(3)),
                tensor_features.RotationQuat(range(3, 7)),
            ]
        )
        dist = tensor_features.get_distance(a, feature, b, feature)
        self.assertTrue(torch.allclose(dist["Coordinate_0"], truth_coord))
        self.assertTrue(torch.allclose(dist["Rotation_1"], truth_rot, atol=1e-3))
