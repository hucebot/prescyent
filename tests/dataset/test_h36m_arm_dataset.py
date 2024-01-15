import warnings

import numpy as np

from tests.custom_test_case import CustomTestCase
from prescyent.dataset import H36MArmDataset, H36MArmDatasetConfig
from prescyent.utils.enums import LearningTypes, RotationRepresentation


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


class H36MArmRotationsDatasetTest(CustomTestCase):
    def test_load_rep6d_none(self):
        try:
            dataset = H36MArmDataset(
                H36MArmDatasetConfig(
                    actions=["directions"],
                    rotation_representation_in=RotationRepresentation.REP6D,
                    rotation_representation_out=None,
                )
            )
            self.assertGreater(len(dataset), 0)
            sample, truth = dataset.test_datasample[0]
            self.assertEqual(len(sample), len(truth))
            self.assertEqual(sample.shape[-1], 9)
            self.assertEqual(truth.shape[-1], 3)
        except NotImplementedError:
            warnings.warn(NO_DATA_WARNING)

    def test_load_rotmat_euler(self):
        try:
            dataset = H36MArmDataset(
                H36MArmDatasetConfig(
                    actions=["directions"],
                    rotation_representation_in=RotationRepresentation.ROTMATRICES,
                    rotation_representation_out=RotationRepresentation.EULER,
                )
            )
            self.assertGreater(len(dataset), 0)
            sample, truth = dataset.test_datasample[0]
            self.assertEqual(len(sample), len(truth))
            self.assertEqual(sample.shape[-1], 12)
            self.assertEqual(truth.shape[-1], 6)
        except NotImplementedError:
            warnings.warn(NO_DATA_WARNING)

    def test_load_quat(self):
        try:
            dataset = H36MArmDataset(
                H36MArmDatasetConfig(
                    actions=["directions"],
                    rotation_representation_in=RotationRepresentation.QUATERNIONS,
                    rotation_representation_out=RotationRepresentation.QUATERNIONS,
                    coordinates_out=[1],
                )
            )
            self.assertGreater(len(dataset), 0)
            sample, truth = dataset.test_datasample[0]
            self.assertEqual(len(sample), len(truth))
            self.assertEqual(sample.shape[-1], 7)
            self.assertEqual(truth.shape[-1], 5)
        except NotImplementedError:
            warnings.warn(NO_DATA_WARNING)
