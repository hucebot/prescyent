"""Predictor class for SARLSTM"""

from prescyent.predictor.lightning.predictor import LightningPredictor
from prescyent.predictor.lightning.models.autoreg.sarlstm.module import TorchModule
from prescyent.predictor.lightning.models.autoreg.sarlstm.config import Config


class Predictor(LightningPredictor):
    """Upper class to train and use a LSTM Model"""

    PREDICTOR_NAME = "SARLSTMPredictor"
    """unique name for this predictror"""
    module_class = TorchModule
    """LightningModule class used in this predictor"""
    config_class = Config
    """PredictorConfig class used in this predictor"""

    def __init__(self, config: Config, skip_build: bool = False):
        super().__init__(config=config, name=self.PREDICTOR_NAME, skip_build=skip_build)
