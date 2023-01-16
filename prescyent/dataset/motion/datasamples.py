from torch import Tensor


class MotionDataSamples():
    x: Tensor
    y: Tensor
    len: int

    def __init__(self, x: Tensor, y: Tensor) -> None:
        self.x = x
        self.y = y
        self.len = self.x.shape[0]
        print("Dataset loaded, length:", self.x.shape[0])

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.len
