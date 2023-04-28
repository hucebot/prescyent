"""Config elements for Linear Pytorch Lightning module usage"""
from prescyent.predictor.lightning.configs.module_config import ModuleConfig


class Config(ModuleConfig):
    """Pydantic Basemodel for Linear Module configuration"""
    input_size: int
    output_size: int
