"""Common config elements for Pytorch Lightning training usage"""
import random
from typing import List, Optional, Union

from pydantic import model_validator

from prescyent.predictor.lightning.configs.optimizer_config import OptimizerConfig
from prescyent.utils.enums.profilers import Profilers


class TrainingConfig(OptimizerConfig):
    """Pydantic Basemodel for Pytorch Lightning Training configuration"""

    max_epochs: int = 100
    """See https://lightning.ai/docs/pytorch/stable/common/trainer.html#max-epochs"""
    max_steps: int = -1
    """See https://lightning.ai/docs/pytorch/stable/common/trainer.html#max-steps"""
    accelerator: str = "auto"
    """See https://lightning.ai/docs/pytorch/stable/common/trainer.html#accelerator"""
    devices: Union[str, int, List[int]] = "auto"
    """See https://lightning.ai/docs/pytorch/stable/common/trainer.html#devices"""
    accumulate_grad_batches: int = 1
    """See https://lightning.ai/docs/pytorch/stable/common/trainer.html#accumulate-grad-batches"""
    log_every_n_steps: int = 1
    """See https://lightning.ai/docs/pytorch/stable/common/trainer.html#log-every-n-steps"""
    gradient_clip_val: Optional[float] = None
    """See https://lightning.ai/docs/pytorch/stable/common/trainer.html#gradient-clip-val"""
    gradient_clip_algorithm: Optional[str] = None
    """See `monitor` in https://lightning.ai/docs/pytorch/stable/common/trainer.html#init"""
    early_stopping_value: str = "Val/loss"
    """The value to be monitored for early stopping
    See https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.EarlyStopping.html#lightning.pytorch.callbacks.EarlyStopping"""
    early_stopping_patience: Optional[int] = None
    """The number of epoch without an improvements before stopping the training
    See https://lightning.ai/docs/pytorch/stable/common/trainer.html#early-stopping-patience"""
    early_stopping_mode: str = "min"
    """The method used to compare the monitored value
    See https://lightning.ai/docs/pytorch/stable/common/trainer.html#early-stopping-mode"""
    use_deterministic_algorithms: bool = True
    """Sets torch.use_deterministic_algorithms and Trainer's deterministic flag to True"""
    use_auto_lr: bool = False
    """If True, lr will be determined by Tuner.lr_finder()
    See https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.tuner.tuning.Tuner.html#lightning.pytorch.tuner.tuning.Tuner.lr_find"""
    seed: Optional[int] = None
    """Seed used during training and any predictor random operation"""
    used_profiler: Optional[Profilers] = None
    """List of profilers to use during training
    See https://lightning.ai/docs/pytorch/stable/tuning/profiler_basic.html"""

    @model_validator(mode="after")
    def training_has_at_least_one_positive_limit(self):
        if self.max_epochs < 0 and self.max_steps < 0:
            raise ValueError(
                "Please set at least one positive limit for the training: "
                '("max_steps" > 0 or "epoch" > 0)'
            )
        return self

    @model_validator(mode="after")
    def generate_random_seed_if_none(self):
        if self.use_deterministic_algorithms and self.seed is None:
            self.seed = random.randint(1, 10**9)
        return self
