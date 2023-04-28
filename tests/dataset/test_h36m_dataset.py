import numpy as np
import torch

from tests.custom_test_case import CustomTestCase
from prescyent.dataset import H36MDataset, H36MDatasetConfig
from prescyent.utils.enums import LearningTypes


class InitH36MDatasetTest(CustomTestCase):

    def test_load_default(self):
        dataset = H36MDataset()
        self.assertGreater(len(dataset), 0)

    def test_load_seq2seq(self):
        dataset = H36MDataset(
            H36MDatasetConfig(actions=['directions'],
                              learning_type=LearningTypes.SEQ2SEQ)
            )
        self.assertGreater(len(dataset), 0)

    def test_load_autoreg(self):
        dataset = H36MDataset(
            H36MDatasetConfig(actions=['directions'],
                              learning_type=LearningTypes.AUTOREG)
            )
        self.assertGreater(len(dataset), 0)
        sample, truth = dataset.test_datasample[0]
        self.assertEqual(len(sample), len(truth))
        np.testing.assert_allclose(sample[1:], truth[:-1], err_msg="thruth and sample differ")
