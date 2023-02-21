"""Config elements for Linear Pytorch Lightning module usage"""
from pydantic import BaseModel


class Config(BaseModel):
    """Pydantic Basemodel for Linear Module configuration"""
    input_size: int
    output_size: int
    identifier: str = "linear"
    model_path: str = "data/models"
