"""Config elements for Seq2Seq Pytorch Lightning module usage"""
from prescyent.predictor.lightning.module_config import ModuleConfig


class Config(ModuleConfig):
    """Pydantic Basemodel for Seq2Seq Module configuration"""
    feature_size: int
    output_size: int
    identifier: str = "seq2seq"
    hidden_size: int = 10
    num_layers: int = 1
