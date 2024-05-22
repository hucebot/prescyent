import shutil

import numpy as np

from tests.custom_test_case import CustomTestCase
from prescyent.dataset import SSTDataset, SSTDatasetConfig
from prescyent.dataset.features import RotationEuler
from prescyent.utils.enums import LearningTypes


class InitSSTDatasetTest(CustomTestCase):
    def test_load_default(self):
        dataset = SSTDataset(SSTDatasetConfig(num_traj=10), load_data_at_init=True)
        self.assertGreater(len(dataset), 0)

    def test_load_seq2seq(self):
        dataset = SSTDataset(
            SSTDatasetConfig(num_traj=10, learning_type=LearningTypes.SEQ2SEQ),
            load_data_at_init=True,
        )
        self.assertGreater(len(dataset), 0)

    def test_load_autoreg(self):
        dataset = SSTDataset(
            SSTDatasetConfig(num_traj=10, learning_type=LearningTypes.AUTOREG),
            load_data_at_init=True,
        )
        self.assertGreater(len(dataset), 0)
        sample, truth = dataset.test_datasample[0]
        self.assertEqual(len(sample), len(truth))
        np.testing.assert_allclose(
            sample[1:], truth[:-1], err_msg="thruth and sample differ"
        )

    def test_load_seq2one(self):
        dataset = SSTDataset(
            SSTDatasetConfig(num_traj=10, learning_type=LearningTypes.SEQ2ONE),
            load_data_at_init=True,
        )
        self.assertGreater(len(dataset), 0)
        _, truth = dataset.test_datasample[0]
        self.assertEqual(1, len(truth))
        self.assertEqual(50, dataset.config.future_size)

    def test_coordinates_1d(self):
        dataset = SSTDataset(
            SSTDatasetConfig(
                num_traj=10,
                out_features=[RotationEuler([0, 1, 2])],
            ),
            load_data_at_init=True,
        )
        self.assertGreater(len(dataset), 0)
        sample, truth = dataset.test_datasample[0]
        self.assertEqual(sample.shape[-1], 7)
        self.assertEqual(truth.shape[-1], 3)
        sample, truth = dataset.train_datasample[0]
        self.assertEqual(sample.shape[-1], 7)
        self.assertEqual(truth.shape[-1], 3)
        sample, truth = dataset.val_datasample[0]
        self.assertEqual(sample.shape[-1], 7)
        self.assertEqual(truth.shape[-1], 3)

    def test_load_from_path(self):
        dataset = SSTDataset(SSTDatasetConfig(num_traj=10), load_data_at_init=True)
        dataset.save_config("tmp/test.json")
        config = dataset._load_config("tmp/test.json")
        SSTDataset("tmp/test.json", load_data_at_init=True)
        shutil.rmtree("tmp", ignore_errors=True)
