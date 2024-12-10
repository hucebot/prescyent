import shutil
import warnings

import numpy as np
import torch
from pydantic import ValidationError

from tests.custom_test_case import CustomTestCase
from prescyent.dataset import AndyDataset, AndyDatasetConfig
from prescyent.dataset.features import CoordinateX, Features, RotationRotMat
from prescyent.utils.enums import LearningTypes


DEFAULT_DATA_PATH = "data/datasets/AndyData-lab-onePerson.hdf5"
NO_DATA_WARNING = "Data for AndyDataset are not installed or found, please refer to the README if you intend to use it"


class InitAndyDatasetTest(CustomTestCase):
    def test_load_default(self):
        try:
            dataset = AndyDataset(
                AndyDatasetConfig(
                    hdf5_path=DEFAULT_DATA_PATH, save_samples_on_disk=False
                ),
            )
            self.assertGreater(len(dataset), 0)
        except FileNotFoundError:
            warnings.warn(NO_DATA_WARNING)

    def test_load_seq2seq(self):
        try:
            dataset = AndyDataset(
                AndyDatasetConfig(
                    hdf5_path=DEFAULT_DATA_PATH,
                    learning_type=LearningTypes.SEQ2SEQ,
                    participants=["909"],
                ),
            )
            self.assertGreater(len(dataset), 0)
        except FileNotFoundError:
            warnings.warn(NO_DATA_WARNING)

    def test_load_autoreg(self):
        try:
            dataset = AndyDataset(
                AndyDatasetConfig(
                    hdf5_path=DEFAULT_DATA_PATH,
                    learning_type=LearningTypes.AUTOREG,
                    participants=["909"],
                ),
            )
            self.assertGreater(len(dataset), 0)
            sample, _, truth = dataset.test_datasample[0]
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
                    hdf5_path=DEFAULT_DATA_PATH,
                    learning_type=LearningTypes.SEQ2ONE,
                    participants=["909"],
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
            dataset = AndyDataset(
                AndyDatasetConfig(
                    hdf5_path=DEFAULT_DATA_PATH,
                    in_features=Features([RotationRotMat(range(9))]),
                    out_features=Features([CoordinateX([0])]),
                    participants=["909"],
                ),
            )
            self.assertGreater(len(dataset), 0)
            sample, _, truth = dataset.test_datasample[0]
            self.assertEqual(sample.shape[-1], 9)
            self.assertEqual(truth.shape[-1], 1)
            sample, _, truth = dataset.train_datasample[0]
            self.assertEqual(sample.shape[-1], 9)
            self.assertEqual(truth.shape[-1], 1)
            sample, _, truth = dataset.val_datasample[0]
            self.assertEqual(sample.shape[-1], 9)
            self.assertEqual(truth.shape[-1], 1)
        except FileNotFoundError:
            warnings.warn(NO_DATA_WARNING)

    def test_load_from_path(self):
        try:
            dataset = AndyDataset(
                AndyDatasetConfig(hdf5_path=DEFAULT_DATA_PATH, participants=["909"]),
            )
            dataset.save_config("tmp/test.json")
            _ = dataset._load_config("tmp/test.json")
            AndyDataset(
                "tmp/test.json",
            )
            shutil.rmtree("tmp", ignore_errors=True)
        except FileNotFoundError:
            warnings.warn(NO_DATA_WARNING)

    def test_load_all_context(self):
        try:
            dataset = AndyDataset(
                config=AndyDatasetConfig(
                    hdf5_path=DEFAULT_DATA_PATH,
                    context_keys=["centerOfMass"],
                    participants=["909"],
                ),
            )
            dataset.prepare_data()
            dataset.setup("test")
            sample, context, truth = next(iter(dataset.test_dataloader()))
            self.assertEqual(
                context["centerOfMass"].shape[0], dataset.config.batch_size
            )
            self.assertEqual(
                context["centerOfMass"].shape[1], dataset.config.history_size
            )
            self.assertEqual(context["centerOfMass"].shape[2], 3)
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
                AndyDatasetConfig(
                    hdf5_path=DEFAULT_DATA_PATH,
                    context_keys=["bad_key", "centerOfMass"],
                )
        except FileNotFoundError:
            warnings.warn(NO_DATA_WARNING)
