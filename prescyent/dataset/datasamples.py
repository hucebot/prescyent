"""Data pair of sample and truth for motion data in ML"""
from torch import Tensor


class MotionDataSamples():
    """Class storing x,y pairs for ML trainings on motion data"""
    sample: Tensor
    truth: Tensor
    len: int

    def __init__(self, sample: Tensor, truth: Tensor) -> None:
        self.sample = sample
        self.truth = truth
        self.len = self.sample.shape[0]

    def __getitem__(self, index):
        return self.sample[index], self.truth[index]

    def __len__(self):
        return self.len
