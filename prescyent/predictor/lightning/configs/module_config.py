"""Config elements for Pytorch Lightning Modules usage"""
from typing import Union, Optional
from pydantic import BaseModel, model_validator

from prescyent.utils.enums import Normalizations
from prescyent.utils.enums import LossFunctions
from prescyent.utils.enums import Profilers


class ModuleConfig(BaseModel):
    """Pydantic Basemodel for Torch Module configuration"""

    version: Optional[int] = None
    input_size: Optional[int] = None
    output_size: Optional[int] = None
    num_dims: Optional[int] = None
    num_points: Optional[int] = None
    input_frequency: Optional[int] = None
    output_frequency: Optional[int] = None
    save_path: str = "data/models"
    dropout_value: Optional[float] = None
    norm_on_last_input: bool = False
    used_norm: Optional[Normalizations] = None
    loss_fn: LossFunctions = LossFunctions.MSELOSS
    do_lipschitz_continuation: bool = False
    used_profiler: Optional[Profilers] = None

    @property
    def feature_size(self):
        return self.num_dims * self.num_points

    @model_validator(mode="after")
    def check_norms_have_requirements(self):
        if self.used_norm in [Normalizations.ALL] and (
            self.input_size is None or self.num_dims is None or self.num_points is None
        ):
            raise ValueError(
                f"{self.used_norm} normalization necessitate a valid "
                "input_size, num_dims and num_points in config"
            )
        elif self.used_norm == Normalizations.SPATIAL and (
            self.num_dims is None or self.num_points is None
        ):
            raise ValueError(
                f"{self.used_norm} normalization necessitate a valid "
                "num_dims and num_points in config"
            )
        elif self.used_norm in [Normalizations.TEMPORAL, Normalizations.BATCH] and (
            self.input_size is None
        ):
            raise ValueError(
                f"{self.used_norm} normalization necessitate a valid "
                "input_size in config"
            )
        return self
