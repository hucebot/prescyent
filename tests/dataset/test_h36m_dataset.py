import warnings

import numpy as np
from pydantic import ValidationError

from tests.custom_test_case import CustomTestCase
from prescyent.dataset import H36MDataset, H36MDatasetConfig
from prescyent.utils.enums import LearningTypes


DEFAULT_DATA_PATH = "data/datasets/h36m.hdf5"
NO_DATA_WARNING = (
    "H36M dataset is not installed, please refer to the README if you intend to use it"
)


class InitH36MDatasetTest(CustomTestCase):
    def test_load_default(self):
        try:
            dataset = H36MDataset(
                H36MDatasetConfig(
                    hdf5_path=DEFAULT_DATA_PATH, save_samples_on_disk=False
                ),
            )
            self.assertGreater(len(dataset), 0)
        except FileNotFoundError:
            warnings.warn(NO_DATA_WARNING)

    def test_load_seq2seq(self):
        try:
            dataset = H36MDataset(
                H36MDatasetConfig(
                    hdf5_path=DEFAULT_DATA_PATH,
                    subjects_train=[],
                    subjects_val=[],
                    actions=["directions"],
                    learning_type=LearningTypes.SEQ2SEQ,
                ),
            )
            self.assertGreater(len(dataset), 0)
        except FileNotFoundError:
            warnings.warn(NO_DATA_WARNING)

    def test_load_autoreg(self):
        try:
            dataset = H36MDataset(
                H36MDatasetConfig(
                    hdf5_path=DEFAULT_DATA_PATH,
                    subjects_train=[],
                    subjects_val=[],
                    actions=["directions"],
                    learning_type=LearningTypes.AUTOREG,
                ),
            )
            self.assertGreater(len(dataset), 0)
            sample, context, truth = dataset.test_datasample[0]
            self.assertEqual(len(sample), len(truth))
            np.testing.assert_allclose(
                sample[1:], truth[:-1], err_msg="thruth and sample differ"
            )
        except FileNotFoundError:
            warnings.warn(NO_DATA_WARNING)

    def test_load_seq2one(self):
        try:
            dataset = H36MDataset(
                H36MDatasetConfig(
                    hdf5_path=DEFAULT_DATA_PATH,
                    subjects_train=[],
                    subjects_val=[],
                    actions=["directions"],
                    learning_type=LearningTypes.SEQ2ONE,
                ),
            )
            self.assertGreater(len(dataset), 0)
            _, _, truth = dataset.test_datasample[0]
            self.assertEqual(1, len(truth))
            self.assertEqual(25, dataset.config.future_size)
        except FileNotFoundError:
            warnings.warn(NO_DATA_WARNING)

    def test_load_bad_context(self):
        try:
            with self.assertRaises(ValidationError):
                H36MDatasetConfig(
                    hdf5_path=DEFAULT_DATA_PATH,
                    context_keys=["any_key"],
                )
        except FileNotFoundError:
            warnings.warn(NO_DATA_WARNING)
