"""Predictor class for SARLSTM"""

from prescyent.predictor.lightning.predictor import LightningPredictor
from prescyent.predictor.lightning.models.autoreg.sarlstm.module import TorchModule
from prescyent.predictor.lightning.models.autoreg.sarlstm.config import Config


class Predictor(LightningPredictor):
    """Upper class to train and use a LSTM Model"""

    PREDICTOR_NAME = "SARLSTMPredictor"
    module_class = TorchModule
    config_class = Config

    def __init__(self, config, skip_build=False):
        super().__init__(config=config, name=self.PREDICTOR_NAME, skip_build=skip_build)
