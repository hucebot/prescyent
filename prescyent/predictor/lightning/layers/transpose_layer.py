"""torch module to perform transpose operation"""
import torch


class TransposeLayer(torch.nn.Module):
    """simple transpose operation as a layer"""

    def __init__(self, dim0, dim1) -> None:
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        return torch.transpose(x, self.dim0, self.dim1)
