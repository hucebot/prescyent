import numpy as np
import torch

from tests.custom_test_case import CustomTestCase
from prescyent.dataset import SineDataset, SineDatasetConfig, LearningTypes


class InitSineDatasetTest(CustomTestCase):

    def test_load_default(self):
        dataset = SineDataset()
        self.assertGreater(len(dataset), 0)

    def test_load_seq2seq(self):
        dataset = SineDataset(
            SineDatasetConfig(learning_type=LearningTypes.SEQ2SEQ)
            )
        self.assertGreater(len(dataset), 0)

    def test_load_autoreg(self):
        dataset = SineDataset(
            SineDatasetConfig(learning_type=LearningTypes.AUTOREG)
            )
        self.assertGreater(len(dataset), 0)
        sample, truth = dataset.test_datasample[0]
        self.assertEqual(len(sample), len(truth))
        np.testing.assert_allclose(sample[1:], truth[:-1], err_msg="thruth and sample differ")
