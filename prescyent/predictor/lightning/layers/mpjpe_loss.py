import torch


class MPJPELoss(torch.nn.modules.loss._Loss):
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(MPJPELoss, self).__init__(size_average, reduce, reduction)

    def forward(self, input_tensor: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        T = target.shape[-1]
        input_tensor_ = input_tensor.reshape(-1, T)
        target_ = target.reshape(-1, T)
        return torch.mean(torch.norm(input_tensor_ - target_, 2, 1))
