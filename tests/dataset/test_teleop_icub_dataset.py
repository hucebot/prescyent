import shutil
import warnings

import numpy as np

from tests.custom_test_case import CustomTestCase
from prescyent.dataset import TeleopIcubDataset, TeleopIcubDatasetConfig
from prescyent.dataset.features import CoordinateXY, Features
from prescyent.utils.enums import LearningTypes


NO_DATA_WARNING = "TeleopIcub dataset is not installed, please refer to the README if you intend to use it"


class InitTeleopIcubDatasetTest(CustomTestCase):
    def test_load_default(self):
        try:
            dataset = TeleopIcubDataset(load_data_at_init=True)
            self.assertGreater(len(dataset), 0)
        except FileNotFoundError:
            warnings.warn(NO_DATA_WARNING)

    def test_load_seq2seq(self):
        try:
            dataset = TeleopIcubDataset(
                TeleopIcubDatasetConfig(learning_type=LearningTypes.SEQ2SEQ),
                load_data_at_init=True,
            )
            self.assertGreater(len(dataset), 0)
        except FileNotFoundError:
            warnings.warn(NO_DATA_WARNING)

    def test_load_autoreg(self):
        try:
            dataset = TeleopIcubDataset(
                TeleopIcubDatasetConfig(learning_type=LearningTypes.AUTOREG),
                load_data_at_init=True,
            )
            self.assertGreater(len(dataset), 0)
            sample, _, truth = dataset.test_datasample[0]
            self.assertEqual(len(sample), len(truth))
            np.testing.assert_allclose(
                sample[1:], truth[:-1], err_msg="truth and sample differ"
            )
        except FileNotFoundError:
            warnings.warn(NO_DATA_WARNING)

    def test_load_seq2one(self):
        try:
            dataset = TeleopIcubDataset(
                TeleopIcubDatasetConfig(learning_type=LearningTypes.SEQ2ONE),
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
            dataset = TeleopIcubDataset(
                TeleopIcubDatasetConfig(
                    out_features=Features([CoordinateXY(range(2))]),
                ),
                load_data_at_init=True,
            )
            self.assertGreater(len(dataset), 0)
            sample, _, truth = dataset.test_datasample[0]
            self.assertEqual(sample.shape[-1], 3)
            self.assertEqual(truth.shape[-1], 2)
            sample, _, truth = dataset.train_datasample[0]
            self.assertEqual(sample.shape[-1], 3)
            self.assertEqual(truth.shape[-1], 2)
            sample, _, truth = dataset.val_datasample[0]
            self.assertEqual(sample.shape[-1], 3)
            self.assertEqual(truth.shape[-1], 2)
        except FileNotFoundError:
            warnings.warn(NO_DATA_WARNING)

    def test_impossible_configs(self):
        try:
            config = TeleopIcubDatasetConfig(future_size=200)
            self.assertRaises(ValueError, TeleopIcubDataset, config, True)
            config = TeleopIcubDatasetConfig(future_size=100, history_size=100)
            self.assertRaises(ValueError, TeleopIcubDataset, config, True)
            config = TeleopIcubDatasetConfig(history_size=100)
            TeleopIcubDataset(config, True)  # this is ok
        except FileNotFoundError:
            warnings.warn(NO_DATA_WARNING)

    def test_load_from_path(self):
        try:
            dataset = TeleopIcubDataset(load_data_at_init=True)
            dataset.save_config("tmp/test.json")
            _ = dataset._load_config("tmp/test.json")
            TeleopIcubDataset("tmp/test.json", load_data_at_init=True)
            shutil.rmtree("tmp", ignore_errors=True)
        except FileNotFoundError:
            warnings.warn(NO_DATA_WARNING)
