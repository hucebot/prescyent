"""Common config elements for Pytorch Lightning training usage"""
from prescyent.base_config import BaseConfig


class OptimizerConfig(BaseConfig):
    """Pydantic Basemodel for Pytorch Lightning Training configuration"""

    lr: float = 1e-3
    """The learning rate used by the Optimizer during training"""
    weight_decay: float = 1e-2
    """The weight decay used by the Optimizer during training"""
    use_scheduler: bool = False  # Used for Scheduler
    """If True, will use a OneCycleLR scheduler for the Learning rate
    See https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html"""
    max_lr: float = 1e-2  # Used for Scheduler
    """Serve as the upper limit for the learning rate if the Scheduler is used"""
