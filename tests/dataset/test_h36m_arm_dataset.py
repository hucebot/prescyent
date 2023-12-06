import warnings

import numpy as np

from tests.custom_test_case import CustomTestCase
from prescyent.dataset.human36m.h36m_arm import H36MArmDataset, H36MArmDatasetConfig
from prescyent.utils.enums import LearningTypes


NO_DATA_WARNING = "H36MArm dataset is not installed, please refer to the README if you intend to use it"


class InitH36MArmDatasetTest(CustomTestCase):
    def test_load_default(self):
        try:
            dataset = H36MArmDataset()
            self.assertGreater(len(dataset), 0)
        except NotImplementedError:
            warnings.warn(NO_DATA_WARNING)

    def test_load_seq2seq(self):
        try:
            dataset = H36MArmDataset(
                H36MArmDatasetConfig(
                    actions=["directions"], learning_type=LearningTypes.SEQ2SEQ
                )
            )
            self.assertGreater(len(dataset), 0)
        except NotImplementedError:
            warnings.warn(NO_DATA_WARNING)

    def test_load_seq2one(self):
        try:
            dataset = H36MArmDataset(
                H36MArmDatasetConfig(
                    actions=["directions"], learning_type=LearningTypes.SEQ2ONE
                )
            )
            self.assertGreater(len(dataset), 0)
            _, truth = dataset.test_datasample[0]
            self.assertEqual(1, len(truth))
            self.assertEqual(1, dataset.config.future_size)
        except NotImplementedError:
            warnings.warn(NO_DATA_WARNING)

    def test_load_autoreg(self):
        try:
            dataset = H36MArmDataset(
                H36MArmDatasetConfig(
                    actions=["directions"], learning_type=LearningTypes.AUTOREG
                )
            )
            self.assertGreater(len(dataset), 0)
            sample, truth = dataset.test_datasample[0]
            self.assertEqual(len(sample), len(truth))
            np.testing.assert_allclose(
                sample[1:], truth[:-1], err_msg="thruth and sample differ"
            )
        except NotImplementedError:
            warnings.warn(NO_DATA_WARNING)
