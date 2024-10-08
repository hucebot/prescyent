import shutil
import warnings

import numpy as np
import torch

from tests.custom_test_case import CustomTestCase
from prescyent.dataset import AndyDataset, AndyDatasetConfig
from prescyent.dataset.features import CoordinateX, Features, RotationRotMat
from prescyent.utils.enums import LearningTypes


NO_DATA_WARNING = "Data for AndyDataset are not installed or found, please refer to the README if you intend to use it"


class InitAndyDatasetTest(CustomTestCase):
    def test_load_default(self):
        try:
            dataset = AndyDataset(
                AndyDatasetConfig(save_samples_on_disk=False), load_data_at_init=True
            )
            self.assertGreater(len(dataset), 0)
        except FileNotFoundError:
            warnings.warn(NO_DATA_WARNING)

    def test_load_seq2seq(self):
        try:
            dataset = AndyDataset(
                AndyDatasetConfig(
                    learning_type=LearningTypes.SEQ2SEQ, participants=["909"]
                ),
                load_data_at_init=True,
            )
            self.assertGreater(len(dataset), 0)
        except FileNotFoundError:
            warnings.warn(NO_DATA_WARNING)

    def test_load_autoreg(self):
        try:
            dataset = AndyDataset(
                AndyDatasetConfig(
                    learning_type=LearningTypes.AUTOREG, participants=["909"]
                ),
                load_data_at_init=True,
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
            dataset = AndyDataset(
                AndyDatasetConfig(
                    learning_type=LearningTypes.SEQ2ONE, participants=["909"]
                ),
                load_data_at_init=True,
            )
            self.assertGreater(len(dataset), 0)
            _, _, truth = dataset.test_datasample[0]
            self.assertEqual(1, len(truth))
            self.assertEqual(10, dataset.config.future_size)
        except FileNotFoundError:
            warnings.warn(NO_DATA_WARNING)

    def test_coordinates_2d(self):
        try:
            dataset = AndyDataset(
                AndyDatasetConfig(
                    in_features=Features([RotationRotMat(range(9))]),
                    out_features=Features([CoordinateX([0])]),
                    participants=["909"],
                ),
                load_data_at_init=True,
            )
            self.assertGreater(len(dataset), 0)
            sample, context, truth = dataset.test_datasample[0]
            self.assertEqual(sample.shape[-1], 9)
            self.assertEqual(truth.shape[-1], 1)
            sample, context, truth = dataset.train_datasample[0]
            self.assertEqual(sample.shape[-1], 9)
            self.assertEqual(truth.shape[-1], 1)
            sample, context, truth = dataset.val_datasample[0]
            self.assertEqual(sample.shape[-1], 9)
            self.assertEqual(truth.shape[-1], 1)
        except FileNotFoundError:
            warnings.warn(NO_DATA_WARNING)

    def test_load_from_path(self):
        try:
            dataset = AndyDataset(
                AndyDatasetConfig(participants=["909"]), load_data_at_init=True
            )
            dataset.save_config("tmp/test.json")
            _ = dataset._load_config("tmp/test.json")
            AndyDataset("tmp/test.json", load_data_at_init=True)
            shutil.rmtree("tmp", ignore_errors=True)
        except FileNotFoundError:
            warnings.warn(NO_DATA_WARNING)
