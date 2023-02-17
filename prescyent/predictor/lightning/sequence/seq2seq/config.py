"""Config elements for Seq2Seq Pytorch Lightning module usage"""
from pydantic import BaseModel


class Config(BaseModel):
    """Pydantic Basemodel for Seq2Seq Module configuration"""
    feature_size: int
    output_size: int
    identifier: str = "seq2seq"
    hidden_size: int = 10
    model_path: str = "data/models"
    num_layers: int = 1
