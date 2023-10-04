"""Config elements for Seq2Seq Pytorch Lightning module usage"""
from prescyent.predictor.lightning.configs.module_config import ModuleConfig


class Config(ModuleConfig):
    """Pydantic Basemodel for Seq2Seq Module configuration"""

    num_points: int
    num_dims: int
    output_size: int
    hidden_size: int = 10
    num_layers: int = 1
