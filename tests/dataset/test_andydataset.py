import shutil

import numpy as np
import torch

from tests.custom_test_case import CustomTestCase
from prescyent.dataset import AndyDataset, AndyDatasetConfig
from prescyent.dataset.features import CoordinateX, RotationRotMat
from prescyent.utils.enums import LearningTypes


class InitAndyDatasetTest(CustomTestCase):
    def test_load_default(self):
        dataset = AndyDataset()
        self.assertGreater(len(dataset), 0)

    def test_load_seq2seq(self):
        dataset = AndyDataset(AndyDatasetConfig(learning_type=LearningTypes.SEQ2SEQ))
        self.assertGreater(len(dataset), 0)

    def test_load_autoreg(self):
        dataset = AndyDataset(AndyDatasetConfig(learning_type=LearningTypes.AUTOREG))
        self.assertGreater(len(dataset), 0)
        sample, truth = dataset.test_datasample[0]
        self.assertEqual(len(sample), len(truth))
        np.testing.assert_allclose(
            sample[1:], truth[:-1], err_msg="thruth and sample differ"
        )

    def test_load_seq2one(self):
        dataset = AndyDataset(AndyDatasetConfig(learning_type=LearningTypes.SEQ2ONE))
        self.assertGreater(len(dataset), 0)
        _, truth = dataset.test_datasample[0]
        self.assertEqual(1, len(truth))
        self.assertEqual(1, dataset.config.future_size)

    def test_coordinates_2d(self):
        dataset = AndyDataset(
            AndyDatasetConfig(
                in_features=[RotationRotMat(range(9))],
                out_features=[CoordinateX([0])],
            )
        )
        self.assertGreater(len(dataset), 0)
        sample, truth = dataset.test_datasample[0]
        self.assertEqual(sample.shape[-1], 9)
        self.assertEqual(truth.shape[-1], 1)
        sample, truth = dataset.train_datasample[0]
        self.assertEqual(sample.shape[-1], 9)
        self.assertEqual(truth.shape[-1], 1)
        sample, truth = dataset.val_datasample[0]
        self.assertEqual(sample.shape[-1], 9)
        self.assertEqual(truth.shape[-1], 1)

    def test_load_from_path(self):
        dataset = AndyDataset()
        dataset.save_config("tmp/test.json")
        config = dataset._load_config("tmp/test.json")
        AndyDataset("tmp/test.json")
        shutil.rmtree("tmp", ignore_errors=True)
