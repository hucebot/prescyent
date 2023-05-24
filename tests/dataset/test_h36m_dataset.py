import warnings

import numpy as np

from tests.custom_test_case import CustomTestCase
from prescyent.dataset import H36MDataset, H36MDatasetConfig
from prescyent.utils.enums import LearningTypes


NO_DATA_WARNING = 'H36M dataset is not installed, please refer to the README if you intend to use it'

class InitH36MDatasetTest(CustomTestCase):

    def test_load_default(self):
        try:
            dataset = H36MDataset()
            self.assertGreater(len(dataset), 0)
        except NotImplementedError:
            warnings.warn(NO_DATA_WARNING)

    def test_load_seq2seq(self):
        try:
            dataset = H36MDataset(
                H36MDatasetConfig(actions=['directions'],
                              learning_type=LearningTypes.SEQ2SEQ)
            )
            self.assertGreater(len(dataset), 0)
        except NotImplementedError:
            warnings.warn(NO_DATA_WARNING)

    def test_load_autoreg(self):
        try:
            dataset = H36MDataset(
                H36MDatasetConfig(actions=['directions'],
                                learning_type=LearningTypes.AUTOREG)
            )
            self.assertGreater(len(dataset), 0)
            sample, truth = dataset.test_datasample[0]
            self.assertEqual(len(sample), len(truth))
            np.testing.assert_allclose(sample[1:], truth[:-1], err_msg="thruth and sample differ")
        except NotImplementedError:
            warnings.warn(NO_DATA_WARNING)
