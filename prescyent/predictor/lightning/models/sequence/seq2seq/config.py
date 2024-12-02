"""Config elements for Seq2Seq Pytorch Lightning module usage"""
from pydantic import Field

from prescyent.predictor.lightning.configs.module_config import ModuleConfig


class Seq2SeqConfig(ModuleConfig):
    """Pydantic Basemodel for Seq2Seq Module configuration"""

    hidden_size: int = 128
    """Hidden size of the GRU layers"""
    num_layers: int = Field(2, gt=0)
    """Num layers in the GRU layers"""
