import warnings

import numpy as np

from tests.custom_test_case import CustomTestCase
from prescyent.dataset import H36MArmDataset, H36MArmDatasetConfig
from prescyent.dataset.features import (
    CoordinateXYZ,
    CoordinateXY,
    CoordinateX,
    Features,
    RotationRep6D,
    RotationQuat,
    RotationEuler,
    RotationRotMat,
)
from prescyent.utils.enums import LearningTypes


DEFAULT_DATA_PATH = "data/datasets/h36m.hdf5"
NO_DATA_WARNING = "H36MArm dataset is not installed, please refer to the README if you intend to use it"


class InitH36MArmDatasetTest(CustomTestCase):
    def test_load_default(self):
        try:
            dataset = H36MArmDataset(
                H36MArmDatasetConfig(
                    hdf5_path=DEFAULT_DATA_PATH, save_samples_on_disk=False
                ),
            )
            self.assertGreater(len(dataset), 0)
        except FileNotFoundError:
            warnings.warn(NO_DATA_WARNING)

    def test_load_seq2seq(self):
        try:
            dataset = H36MArmDataset(
                H36MArmDatasetConfig(
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

    def test_load_seq2one(self):
        try:
            dataset = H36MArmDataset(
                H36MArmDatasetConfig(
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

    def test_load_autoreg(self):
        try:
            dataset = H36MArmDataset(
                H36MArmDatasetConfig(
                    hdf5_path=DEFAULT_DATA_PATH,
                    subjects_train=[],
                    subjects_val=[],
                    actions=["directions"],
                    learning_type=LearningTypes.AUTOREG,
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


class H36MArmRotationsDatasetTest(CustomTestCase):
    def test_inverse_inputs(self):
        try:
            dataset = H36MArmDataset(
                H36MArmDatasetConfig(
                    hdf5_path=DEFAULT_DATA_PATH,
                    subjects_train=[],
                    subjects_val=[],
                    actions=["directions"],
                    in_features=Features(
                        [
                            CoordinateXYZ(list(range(3))),
                            RotationRep6D(list(range(3, 9))),
                        ]
                    ),
                    out_features=Features(
                        [
                            CoordinateXYZ(list(range(6, 9))),
                            RotationRep6D(list(range(6))),
                        ]
                    ),
                ),
            )
            self.assertGreater(len(dataset), 0)
            _, _, truth = dataset.test_datasample[0]
            sample, _, _ = dataset.test_datasample[dataset.config.history_size]
            self.assertTrue(
                all(
                    [
                        sample.shape[i] == truth.shape[i]
                        for i in range(len(sample.shape))
                    ]
                )
            )
            self.assertEqual(sample.shape[-1], 9)
            self.assertEqual(truth.shape[-1], 9)
            np.testing.assert_allclose(sample[..., [0, 1, 2]], truth[..., [6, 7, 8]])
        except FileNotFoundError:
            warnings.warn(NO_DATA_WARNING)

    def test_load_rep6d_none(self):
        try:
            dataset = H36MArmDataset(
                H36MArmDatasetConfig(
                    hdf5_path=DEFAULT_DATA_PATH,
                    subjects_train=[],
                    subjects_val=[],
                    actions=["directions"],
                    in_features=Features(
                        [
                            CoordinateXYZ(range(3)),
                            RotationRep6D([3, 4, 5, 6, 7, 8]),
                        ]
                    ),
                    out_features=Features([CoordinateXYZ(range(3))]),
                ),
            )
            self.assertGreater(len(dataset), 0)
            sample, _, truth = dataset.test_datasample[0]
            self.assertEqual(len(sample), len(truth))
            self.assertEqual(sample.shape[-1], 9)
            self.assertEqual(truth.shape[-1], 3)
        except FileNotFoundError:
            warnings.warn(NO_DATA_WARNING)

    def test_load_rotmat_euler(self):
        try:
            dataset = H36MArmDataset(
                H36MArmDatasetConfig(
                    hdf5_path=DEFAULT_DATA_PATH,
                    subjects_train=[],
                    subjects_val=[],
                    actions=["directions"],
                    in_features=Features(
                        [CoordinateXYZ(range(3)), RotationRotMat(range(3, 12))]
                    ),
                    out_features=Features(
                        [CoordinateXY(range(2)), RotationEuler(range(2, 5))]
                    ),
                ),
            )
            self.assertGreater(len(dataset), 0)
            sample, _, truth = dataset.test_datasample[0]
            self.assertEqual(len(sample), len(truth))
            self.assertEqual(sample.shape[-1], 12)
            self.assertEqual(truth.shape[-1], 5)
        except FileNotFoundError:
            warnings.warn(NO_DATA_WARNING)

    def test_load_quat(self):
        try:
            dataset = H36MArmDataset(
                H36MArmDatasetConfig(
                    hdf5_path=DEFAULT_DATA_PATH,
                    subjects_train=[],
                    subjects_val=[],
                    actions=["directions"],
                    in_features=Features(
                        [
                            CoordinateXYZ(range(3)),
                            RotationRep6D(range(3, 9)),
                        ]
                    ),
                    out_features=Features(
                        [CoordinateX(range(1)), RotationQuat(range(1, 5))]
                    ),
                ),
            )
            self.assertGreater(len(dataset), 0)
            sample, _, truth = dataset.test_datasample[0]
            self.assertEqual(len(sample), len(truth))
            self.assertEqual(sample.shape[-1], 9)
            self.assertEqual(truth.shape[-1], 5)
        except FileNotFoundError:
            warnings.warn(NO_DATA_WARNING)
