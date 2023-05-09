"""Common config elements for Pytorch Lightning training usage"""
from typing import Union

from pydantic import root_validator

from prescyent.predictor.lightning.configs.optimizer_config import OptimizerConfig


class TrainingConfig(OptimizerConfig):
    """Pydantic Basemodel for Pytorch Lightning Training configuration"""
    epoch: int = 100
    max_steps: int = -1
    accelerator: str = "auto"
    devices: Union[str, int] = "auto"
    accumulate_grad_batches: int = 1
    seed: Union[None, int] = 5
    log_every_n_steps: int = 1
    use_auto_lr: bool = False
    use_deterministic_algorithms: bool = True

    @root_validator
    def training_has_at_least_one_positive_limit(cls, values):
        epoch, max_steps = values.get('epoch'), values.get('max_steps')
        if epoch > 0 and max_steps > 0:
            raise ValueError('Please set at least one positive limit for the training: '
                             '("max_steps" > 0 or "epoch" > 0)')
        return values
