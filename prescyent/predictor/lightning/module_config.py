"""Config elements for Pytorch Lightning Modules usage"""
from pydantic import BaseModel


class ModuleConfig(BaseModel):
    """Pydantic Basemodel for Seq2Seq Module configuration"""
    model_path: str = "data/models"
    do_normalization: bool = False
