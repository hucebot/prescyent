import shutil
import warnings

import numpy as np
from pydantic import ValidationError

from tests.custom_test_case import CustomTestCase
from prescyent.dataset import TeleopIcubDataset, TeleopIcubDatasetConfig
from prescyent.dataset.features import CoordinateXY, Features
from prescyent.utils.enums import LearningTypes


DEFAULT_DATA_PATH = "data/datasets/AndyData-lab-prescientTeleopICub.hdf5"
NO_DATA_WARNING = "TeleopIcub dataset is not installed, please refer to the README if you intend to use it"


class InitTeleopIcubDatasetTest(CustomTestCase):
    def test_load_default(self):
        try:
            dataset = TeleopIcubDataset(
                TeleopIcubDatasetConfig(hdf5_path=DEFAULT_DATA_PATH)
            )
            self.assertGreater(len(dataset), 0)
            sample, context, truth = dataset.test_datasample[0]
            self.assertEqual(context, {})
            self.assertEqual(sample.shape[1], dataset.config.history_size)
            self.assertEqual(sample.shape[2], dataset.config.num_in_points)
            self.assertEqual(sample.shape[3], dataset.config.num_in_dims)
            self.assertEqual(truth.shape[1], dataset.config.future_size)
            self.assertEqual(truth.shape[2], dataset.config.num_out_points)
            self.assertEqual(truth.shape[3], dataset.config.num_out_dims)
        except FileNotFoundError:
            warnings.warn(NO_DATA_WARNING)

    def test_load_seq2seq(self):
        try:
            dataset = TeleopIcubDataset(
                TeleopIcubDatasetConfig(
                    hdf5_path=DEFAULT_DATA_PATH,
                    subsets=["BottleTable"],
                    learning_type=LearningTypes.SEQ2SEQ,
                ),
            )
            self.assertGreater(len(dataset), 0)
        except FileNotFoundError:
            warnings.warn(NO_DATA_WARNING)

    def test_load_autoreg(self):
        try:
            dataset = TeleopIcubDataset(
                TeleopIcubDatasetConfig(
                    hdf5_path=DEFAULT_DATA_PATH,
                    subsets=["BottleTable"],
                    learning_type=LearningTypes.AUTOREG,
                ),
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
                TeleopIcubDatasetConfig(
                    hdf5_path=DEFAULT_DATA_PATH,
                    subsets=["BottleTable"],
                    learning_type=LearningTypes.SEQ2ONE,
                ),
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
                    hdf5_path=DEFAULT_DATA_PATH,
                    subsets=["BottleTable"],
                    out_features=Features([CoordinateXY(range(2))]),
                ),
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
            config = TeleopIcubDatasetConfig(
                hdf5_path=DEFAULT_DATA_PATH, future_size=200
            )
            dataset = TeleopIcubDataset(config)
            self.assertRaises(ValueError, dataset.setup)
            config = TeleopIcubDatasetConfig(
                hdf5_path=DEFAULT_DATA_PATH, future_size=100, history_size=100
            )
            dataset = TeleopIcubDataset(config)
            self.assertRaises(ValueError, dataset.setup)
            config = TeleopIcubDatasetConfig(
                hdf5_path=DEFAULT_DATA_PATH, history_size=100
            )
            TeleopIcubDataset(config)  # this is ok
        except FileNotFoundError:
            warnings.warn(NO_DATA_WARNING)

    def test_load_from_path(self):
        try:
            dataset = TeleopIcubDataset(
                config=TeleopIcubDatasetConfig(hdf5_path=DEFAULT_DATA_PATH)
            )
            dataset.save_config("tmp/test.json")
            _ = dataset._load_config("tmp/test.json")
            TeleopIcubDataset(
                "tmp/test.json",
            )
            shutil.rmtree("tmp", ignore_errors=True)
        except FileNotFoundError:
            warnings.warn(NO_DATA_WARNING)

    def test_load_all_context(self):
        try:
            dataset = TeleopIcubDataset(
                config=TeleopIcubDatasetConfig(
                    hdf5_path=DEFAULT_DATA_PATH,
                    subsets=["BottleTable"],
                    context_keys=["center_of_mass", "icub_dof"],
                ),
            )
            sample, context, truth = next(iter(dataset.test_dataloader()))
            self.assertEqual(
                context["center_of_mass"].shape[0], dataset.config.batch_size
            )
            self.assertEqual(
                context["center_of_mass"].shape[1], dataset.config.history_size
            )
            self.assertEqual(context["center_of_mass"].shape[2], 3)
            self.assertEqual(context["icub_dof"].shape[0], dataset.config.batch_size)
            self.assertEqual(context["icub_dof"].shape[1], dataset.config.history_size)
            self.assertEqual(context["icub_dof"].shape[2], 32)
            self.assertEqual(sample.shape[0], dataset.config.batch_size)
            self.assertEqual(sample.shape[1], dataset.config.history_size)
            self.assertEqual(sample.shape[2], dataset.config.num_in_points)
            self.assertEqual(sample.shape[3], dataset.config.num_in_dims)
            self.assertEqual(truth.shape[0], dataset.config.batch_size)
            self.assertEqual(truth.shape[1], dataset.config.future_size)
            self.assertEqual(truth.shape[2], dataset.config.num_out_points)
            self.assertEqual(truth.shape[3], dataset.config.num_out_dims)
        except FileNotFoundError:
            warnings.warn(NO_DATA_WARNING)

    def test_load_bad_context(self):
        try:
            with self.assertRaises(ValidationError):
                TeleopIcubDatasetConfig(
                    hdf5_path=DEFAULT_DATA_PATH,
                    context_keys=["bad_key", "icub_dof"],
                )
        except FileNotFoundError:
            warnings.warn(NO_DATA_WARNING)
