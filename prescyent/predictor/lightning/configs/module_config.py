"""Config elements for Pytorch Lightning Modules usage"""
from typing import Optional
from pydantic import BaseModel, model_validator

from prescyent.utils.enums import Normalizations
from prescyent.utils.enums import LossFunctions
from prescyent.utils.enums import Profilers
from prescyent.dataset.config import MotionDatasetConfig


class ModuleConfig(BaseModel):
    """Pydantic Basemodel for Torch Module configuration"""

    dataset_config: MotionDatasetConfig
    version: Optional[int] = None
    save_path: str = "data/models"
    dropout_value: Optional[float] = None
    norm_on_last_input: bool = False
    used_norm: Optional[Normalizations] = None
    loss_fn: Optional[LossFunctions] = None
    do_lipschitz_continuation: bool = False
    used_profiler: Optional[Profilers] = None

    @property
    def in_sequence_size(self):
        return self.dataset_config.history_size

    @property
    def out_sequence_size(self):
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
        elif self.used_norm in [Normalizations.TEMPORAL, Normalizations.BATCH] and (
            self.in_sequence_size is None
        ):
            raise ValueError(
                f"{self.used_norm} normalization necessitate a valid "
                "in_sequence_size in config"
            )
        return self
