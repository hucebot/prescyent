"""Common config elements for Pytorch Lightning training usage"""
from pydantic import BaseModel


class OptimizerConfig(BaseModel):
    """Pydantic Basemodel for Pytorch Lightning Training configuration"""

    lr: float = 1e-3
    weight_decay: float = 1e-2
    max_lr: float = 1e-2  # Used for Scheduler
    use_scheduler: bool = False  # Used for Scheduler
