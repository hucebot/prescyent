import shutil
from pathlib import Path

import numpy as np

from tests.custom_test_case import CustomTestCase
from prescyent.dataset import TeleopIcubDataset, TeleopIcubDatasetConfig


class InitTeleopIcubDatasetTest(CustomTestCase):

    def test_load(self):
        dataset = TeleopIcubDataset()
        self.assertGreater(len(dataset), 0)

    # def test_download(self):
    #     tmp_data_path = Path("tmp")
    #     config = TeleopIcubDatasetConfig(data_path=str(tmp_data_path / "icub_data"))
    #     dataset = TeleopIcubDataset(config)
    #     self.assertEqual(len(dataset), 20)
    #     shutil.rmtree(str(Path("tmp")))


class TeleopIcubDatasetTest(CustomTestCase):

    def test_scale(self):
        dataset = TeleopIcubDataset()
        sample = dataset.episodes.train[0].tensor
        nom_sample = dataset.scale(sample)
        # the method is deterministic
        np.testing.assert_allclose(dataset.scale(sample), nom_sample)
        unorned_sample = dataset.unscale(nom_sample)
        # the reverse function is correct
        np.testing.assert_allclose(sample, unorned_sample)
