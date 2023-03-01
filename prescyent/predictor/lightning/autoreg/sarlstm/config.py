"""Config elements for SARLSTM Lightning module usage"""
from prescyent.predictor.lightning.module_config import ModuleConfig


class Config(ModuleConfig):
    """Pydantic Basemodel for SARLSTM configuration"""
    feature_size: int
    hidden_size: int = 10
