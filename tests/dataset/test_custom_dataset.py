import shutil

import numpy as np

from tests.custom_test_case import CustomTestCase
from prescyent.dataset import SSTDataset, SSTDatasetConfig
from prescyent.dataset.features import Features, RotationQuat
from prescyent.utils.enums import LearningTypes


class InitCustomDatasetTest(CustomTestCase):
    def test_init_bad_shapes(self):
        pass
