"""Config elements for Linear Pytorch Lightning module usage"""
from typing import List
from prescyent.predictor.lightning.module_config import ModuleConfig


class Config(ModuleConfig):
    """Pydantic Basemodel for MLP Module configuration"""
    input_size: int
    output_size: int
    hidden_size : List[int] = [64, 64]
    activation: str = "ReLu"
    # we could add :
    # lipchiz normalisation
    # batchnorm
    # dropout
