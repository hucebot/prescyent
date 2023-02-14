"""Config elements for LSTM Pytorch Lightning module usage"""
from pydantic import BaseModel


class Config(BaseModel):
    """Pydantic Basemodel for LSTM Module configuration"""
    feature_size: int
    output_size: int
    identifier: str = "lstm"
    hidden_size: int = 10
    model_path: str = "data/models"
    num_layers: int = 1
