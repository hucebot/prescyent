"""Config elements for MLP Pytorch Lightning module usage"""
from prescyent.predictor.lightning.configs.module_config import ModuleConfig


class Config(ModuleConfig):
    """Pydantic Basemodel for MLP Module configuration"""

    input_size: int
    output_size: int
    hidden_size: int = 64
    num_layers: int = 2
    activation: str = "ReLu"
