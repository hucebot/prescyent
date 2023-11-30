import shutil

import numpy as np
import torch

from tests.custom_test_case import CustomTestCase
from prescyent.dataset import TeleopIcubDataset, TeleopIcubDatasetConfig
from prescyent.utils.enums import LearningTypes


class InitTeleopIcubDatasetTest(CustomTestCase):
    def test_load_default(self):
        dataset = TeleopIcubDataset()
        self.assertGreater(len(dataset), 0)

    def test_load_seq2seq(self):
        dataset = TeleopIcubDataset(
            TeleopIcubDatasetConfig(learning_type=LearningTypes.SEQ2SEQ)
        )
        self.assertGreater(len(dataset), 0)

    def test_load_autoreg(self):
        dataset = TeleopIcubDataset(
            TeleopIcubDatasetConfig(learning_type=LearningTypes.AUTOREG)
        )
        self.assertGreater(len(dataset), 0)
        sample, truth = dataset.test_datasample[0]
        self.assertEqual(len(sample), len(truth))
        np.testing.assert_allclose(
            sample[1:], truth[:-1], err_msg="thruth and sample differ"
        )

    def test_load_seq2one(self):
        dataset = TeleopIcubDataset(
            TeleopIcubDatasetConfig(
                actions=["directions"], learning_type=LearningTypes.SEQ2ONE
            )
        )
        self.assertGreater(len(dataset), 0)
        _, truth = dataset.test_datasample[0]
        self.assertEqual(1, len(truth))
        self.assertEqual(1, dataset.future_size)

    def test_impossible_configs(self):
        config = TeleopIcubDatasetConfig(future_size=200)
        self.assertRaises(ValueError, TeleopIcubDataset, config)
        config = TeleopIcubDatasetConfig(future_size=100, history_size=100)
        self.assertRaises(ValueError, TeleopIcubDataset, config)
        config = TeleopIcubDatasetConfig(history_size=100)
        TeleopIcubDataset(config)  # this is ok

    def test_load_from_path(self):
        dataset = TeleopIcubDataset()
        dataset.save_config("tmp/test.json")
        dataset._load_config("tmp/test.json")
        TeleopIcubDataset("tmp/test.json")
        shutil.rmtree("tmp", ignore_errors=True)
