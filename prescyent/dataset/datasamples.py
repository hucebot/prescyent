"""Data pair of sample and truth for motion data in ML"""
from torch import Tensor

from prescyent.utils.logger import logger, DATASET


class MotionDataSamples():
    """Class storing x,y pairs for ML trainings on motion data"""
    sample: Tensor
    truth: Tensor
    len: int

    def __init__(self, sample: Tensor, truth: Tensor) -> None:
        self.sample = sample
        self.truth = truth
        self.len = self.sample.shape[0]
        logger.info("Dataset loaded, length %d", self.sample.shape[0],
                    group=DATASET)

    def __getitem__(self, index):
        return self.sample[index], self.truth[index]

    def __len__(self):
        return self.len
