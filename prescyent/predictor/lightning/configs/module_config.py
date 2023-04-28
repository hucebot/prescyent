"""Config elements for Pytorch Lightning Modules usage"""
from typing import Union, Optional
from pydantic import BaseModel, root_validator

from prescyent.utils.enums import Normalizations
from prescyent.utils.enums import LossFunctions
from prescyent.utils.enums import Profilers


class ModuleConfig(BaseModel):
    """Pydantic Basemodel for Seq2Seq Module configuration"""
    input_size: Optional[int]
    output_size: Optional[int]
    num_dims: Optional[int]
    num_points: Optional[int]
    model_path: str = "data/models"
    dropout_value: Union[None, float] = None
    norm_on_last_input: bool = False
    used_norm: Union[None, Normalizations] = None
    loss_fn: LossFunctions = "mpjpeloss"
    used_profiler: Union[None, Profilers] = None

    @root_validator
    def check_norms_have_requirements(cls, values):
        used_norm = values.get('used_norm')
        if used_norm == Normalizations.ALL and (
                values.get('input_size', None) is None
                or values.get('num_dims', None) is None
                or values.get('num_points', None) is None):
            raise ValueError(f'{used_norm} normalization necessitate a valid '
                             'input_size, num_dims and num_points in config')
        elif used_norm == Normalizations.SPACE_NORM and (
                values.get('num_dims', None) is None
                or values.get('num_points', None) is None):
            raise ValueError(f'{used_norm} normalization necessitate a valid '
                             'num_dims and num_points in config')
        elif used_norm in [Normalizations.BATCH_NORM, Normalizations.TIME_NORM] and (
                values.get('input_size', None) is None):
            raise ValueError(f'{used_norm} normalization necessitate a valid '
                             'input_size in config')
        return values
