"""Predictor class for SARLSTM"""

from prescyent.predictor.lightning.predictor import LightningPredictor
from .module import SARLSTMTorchModule
from .config import SARLSTMConfig


class SARLSTMPredictor(LightningPredictor):
    """Upper class to train and use a LSTM Model"""

    PREDICTOR_NAME = "SARLSTMPredictor"
    """unique name for this predictror"""
    module_class = SARLSTMTorchModule
    """LightningModule class used in this predictor"""
    config_class = SARLSTMConfig
    """PredictorConfig class used in this predictor"""

    def __init__(self, config: SARLSTMConfig, skip_build: bool = False):
        super().__init__(config=config, name=self.PREDICTOR_NAME, skip_build=skip_build)
