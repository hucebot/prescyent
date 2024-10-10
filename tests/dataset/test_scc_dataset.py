import shutil

import numpy as np

from tests.custom_test_case import CustomTestCase
from prescyent.dataset import SCCDataset, SCCDatasetConfig
from prescyent.dataset.features import CoordinateX, Features
from prescyent.utils.enums import LearningTypes


class InitSCCDatasetTest(CustomTestCase):
    def test_load_default(self):
        dataset = SCCDataset(load_data_at_init=True)
        self.assertGreater(len(dataset), 0)

    def test_load_seq2seq(self):
        dataset = SCCDataset(
            SCCDatasetConfig(learning_type=LearningTypes.SEQ2SEQ),
            load_data_at_init=True,
        )
        self.assertGreater(len(dataset), 0)

    def test_load_autoreg(self):
        dataset = SCCDataset(
            SCCDatasetConfig(learning_type=LearningTypes.AUTOREG),
            load_data_at_init=True,
        )
        self.assertGreater(len(dataset), 0)
        sample, context, truth = dataset.test_datasample[0]
        self.assertEqual(len(sample), len(truth))
        np.testing.assert_allclose(
            sample[1:], truth[:-1], err_msg="thruth and sample differ"
        )

    def test_load_seq2one(self):
        dataset = SCCDataset(
            SCCDatasetConfig(learning_type=LearningTypes.SEQ2ONE),
            load_data_at_init=True,
        )
        self.assertGreater(len(dataset), 0)
        _, _, truth = dataset.test_datasample[0]
        self.assertEqual(1, len(truth))
        self.assertEqual(10, dataset.config.future_size)

    def test_coordinates_1d(self):
        dataset = SCCDataset(
            SCCDatasetConfig(
                out_features=Features([CoordinateX([0])]),
            ),
            load_data_at_init=True,
        )
        self.assertGreater(len(dataset), 0)
        sample, _, truth = dataset.test_datasample[0]
        self.assertEqual(sample.shape[-1], 2)
        self.assertEqual(truth.shape[-1], 1)
        sample, _, truth = dataset.train_datasample[0]
        self.assertEqual(sample.shape[-1], 2)
        self.assertEqual(truth.shape[-1], 1)
        sample, _, truth = dataset.val_datasample[0]
        self.assertEqual(sample.shape[-1], 2)
        self.assertEqual(truth.shape[-1], 1)

    def test_load_from_path(self):
        dataset = SCCDataset(load_data_at_init=True)
        dataset.save_config("tmp/test.json")
        _ = dataset._load_config("tmp/test.json")
        SCCDataset("tmp/test.json", load_data_at_init=True)
        shutil.rmtree("tmp", ignore_errors=True)
