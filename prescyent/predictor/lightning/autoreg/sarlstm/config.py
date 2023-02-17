"""Config elements for SARLSTM Lightning module usage"""
from pydantic import BaseModel


class Config(BaseModel):
    """Pydantic Basemodel for SARLSTM configuration"""
    feature_size: int
    identifier: str = "sarlstm"
    hidden_size: int = 10
    model_path: str = "data/models"
