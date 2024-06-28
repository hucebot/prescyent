"""Config elements for Pytorch Lightning Modules usage"""
from typing import Optional
from pydantic import model_validator

from prescyent.base_config import BaseConfig
from prescyent.utils.enums import (
    Normalizations,
    LearningTypes,
    LossFunctions,
    Profilers,
)
from prescyent.dataset.config import MotionDatasetConfig


class ModuleConfig(BaseConfig):
    """Pydantic Basemodel for Torch Module configuration"""

    dataset_config: MotionDatasetConfig
    """We use the MotionDatasetConfig as a source of information for the shapes of our model"""
    name: Optional[str] = None
    """The name of the predictor.
    If None, default to the class value.
    WARNING, If you override default value, AutoPredictor won't be able to load your Predictor"""
    version: Optional[int] = None
    """A version number for this instance of the predictor.
    If None, we'll use TensorBoardLogger logic to aquire a version number from the log path"""
    save_path: str = "data/models"
    """Directory where the model will log and save"""
    used_profiler: Optional[Profilers] = None
    """List of profilers to use during training
    See https://lightning.ai/docs/pytorch/stable/tuning/profiler_basic.html"""
    loss_fn: Optional[LossFunctions] = None
    """Define what loss function will be used to train your model"""
    # Torch Module infos
    do_lipschitz_continuation: bool = False
    """If True, we'll apply Spectral Normalization to every layer of the model"""
    dropout_value: Optional[float] = None
    """Value for the torch Dropout layer as one of the first steps of the forward method of the torch module,
    Default to None results is no Dropout layer
    See https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html"""
    used_norm: Optional[Normalizations] = None
    """Apply given Normalization as the first layer of the model"""
    norm_on_last_input: Optional[bool] = False
    """If True, we'll make the whole input that is fed to the model relative to its last frame,
    It also makes the model's output relative to this frame"""
    deriv_output: Optional[bool] = False
    """If True, the model's output is relative to the last frame of the input"""

    @property
    def in_sequence_size(self):
        return self.dataset_config.history_size

    @property
    def out_sequence_size(self):
        if self.dataset_config.learning_type == LearningTypes.SEQ2ONE:
            # If we predict one value instead of a sequence
            return 1
        return self.dataset_config.future_size

    @property
    def in_points_dims(self):
        return self.dataset_config.num_in_dims * self.dataset_config.num_in_points

    @property
    def out_points_dims(self):
        return self.dataset_config.num_out_dims * self.dataset_config.num_out_points

    @model_validator(mode="after")
    def check_norms_have_requirements(self):
        if self.used_norm in [Normalizations.ALL] and (
            self.in_sequence_size is None
            or self.dataset_config.in_dims is None
            or self.dataset_config.in_points is None
        ):
            raise ValueError(
                f"{self.used_norm} normalization necessitate a valid "
                "in_sequence_size, in_dims and in_points in config"
            )
        elif self.used_norm == Normalizations.SPATIAL and (
            self.dataset_config.in_dims is None or self.dataset_config.in_points is None
        ):
            raise ValueError(
                f"{self.used_norm} normalization necessitate a valid "
                "in_dims and in_points in config"
            )
        elif self.used_norm in [
            Normalizations.TEMPORAL,
            Normalizations.BATCH,
        ] and (self.in_sequence_size is None):
            raise ValueError(
                f"{self.used_norm} normalization necessitate a valid "
                "in_sequence_size in config"
            )
        return self
