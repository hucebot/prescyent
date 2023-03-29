"""Config elements for Pytorch Lightning Modules usage"""
from pydantic import BaseModel


class ModuleConfig(BaseModel):
    """Pydantic Basemodel for Seq2Seq Module configuration"""
    model_path: str = "data/models"
    norm_on_last_input: bool = False
    do_layernorm: bool = False  # TODO
    do_batchnorm: bool = False  # TODO
    criterion: str = "mseloss"
