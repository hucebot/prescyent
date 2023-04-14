"""Config elements for Pytorch Lightning Modules usage"""
from typing import Union, Optional
from pydantic import BaseModel


class ModuleConfig(BaseModel):
    """Pydantic Basemodel for Seq2Seq Module configuration"""
    input_size: int
    output_size: int
    num_dims: Optional[int]
    num_points: Optional[int]
    model_path: str = "data/models"
    norm_on_last_input: bool = False
    do_layernorm: bool = False
    do_batchnorm: bool = False
    dropout_value: Union[None, float] = None
    criterion: str = "mpjpeloss"
    used_profiler: Optional[str] = None
