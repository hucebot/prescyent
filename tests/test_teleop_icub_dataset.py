import numpy as np
import torch

from tests.custom_test_case import CustomTestCase
from prescyent.dataset import TeleopIcubDataset, TeleopIcubDatasetConfig


class InitTeleopIcubDatasetTest(CustomTestCase):

    def test_load(self):
        dataset = TeleopIcubDataset()
        self.assertGreater(len(dataset), 0)


class TeleopIcubDatasetTest(CustomTestCase):

    def test_scale(self):
        dataset = TeleopIcubDataset()
        sample = torch.FloatTensor(dataset.trajectories.train[0].tensor)
        nom_sample = dataset.scale(sample)
        # the method is deterministic
        np.testing.assert_allclose(dataset.scale(sample), nom_sample)
        unorned_sample = dataset.unscale(nom_sample)
        # the reverse function is correct
        np.testing.assert_allclose(sample, unorned_sample, rtol=1e-5)
