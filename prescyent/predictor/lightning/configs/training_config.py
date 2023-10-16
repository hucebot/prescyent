"""Common config elements for Pytorch Lightning training usage"""
import random
from typing import Union, Optional

from pydantic import model_validator, field_validator

from prescyent.predictor.lightning.configs.optimizer_config import OptimizerConfig


class TrainingConfig(OptimizerConfig):
    """Pydantic Basemodel for Pytorch Lightning Training configuration"""

    epoch: int = 100
    max_steps: int = -1
    accelerator: str = "auto"
    devices: Union[str, int] = "auto"
    accumulate_grad_batches: int = 1
    seed: Optional[int] = None
    log_every_n_steps: int = 1
    use_auto_lr: bool = False
    use_deterministic_algorithms: bool = True
    early_stopping_value: str = "Val/loss"
    early_stopping_patience: Optional[int] = None
    early_stopping_mode: str = "min"

    @model_validator(mode="after")
    def training_has_at_least_one_positive_limit(self):
        if self.epoch < 0 and self.max_steps < 0:
            raise ValueError(
                "Please set at least one positive limit for the training: "
                '("max_steps" > 0 or "epoch" > 0)'
            )
        return self

    @field_validator("seed")
    @classmethod
    def generate_random_seed_if_none(cls, v: int):
        if v is None:
            return random.randint(1, 10**9)
