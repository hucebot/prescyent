"""Common config elements for Pytorch Lightning training usage"""
from typing import Union

from prescyent.predictor.lightning.optimizer_config import OptimizerConfig


class TrainingConfig(OptimizerConfig):
    """Pydantic Basemodel for Pytorch Lightning Training configuration"""
    epoch: int = 100
    max_steps: int = -1
    accelerator: str = "auto"
    devices: Union[str, int] = "auto"
