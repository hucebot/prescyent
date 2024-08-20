"""Config elements for Pytorch Lightning Modules usage"""
from typing import Optional
from pydantic import model_validator

from prescyent.predictor.config import PredictorConfig
from prescyent.utils.enums import (
    LearningTypes,
    LossFunctions,
    Profilers,
)
from prescyent.dataset.config import MotionDatasetConfig


class ModuleConfig(PredictorConfig):
    """Pydantic Basemodel for Torch Module configuration"""

    used_profiler: Optional[Profilers] = None
    """List of profilers to use during training
    See https://lightning.ai/docs/pytorch/stable/tuning/profiler_basic.html"""
    loss_fn: Optional[LossFunctions] = None
    """Define what loss function will be used to train your model"""
    do_lipschitz_continuation: bool = False
    """If True, we'll apply Spectral Normalization to every layer of the model"""
    dropout_value: Optional[float] = None
    """Value for the torch Dropout layer as one of the first steps of the forward method of the torch module,
    Default to None results is no Dropout layer
    See https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html"""
    norm_on_last_input: Optional[bool] = False
    """If True, we'll make the whole input that is fed to the model relative to its last frame,
    It also makes the model's output relative to this frame"""
    deriv_output: Optional[bool] = False
    """If True, the model's output is relative to the last frame of the input"""
