from torch import Tensor

from prescyent.logger import logger, DATASET


class MotionDataSamples():
    x: Tensor
    y: Tensor
    len: int

    def __init__(self, x: Tensor, y: Tensor) -> None:
        self.x = x
        self.y = y
        self.len = self.x.shape[0]
        logger.info("Dataset loaded, length %d", self.x.shape[0],
                    group=DATASET)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.len
