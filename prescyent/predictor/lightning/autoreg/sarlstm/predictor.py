"""Pytorch module et Lightning module for LSTMs"""

from prescyent.predictor.lightning.autoreg.predictor import AutoRegPredictor
from prescyent.predictor.lightning.autoreg.sarlstm.module import LightningModule
from prescyent.predictor.lightning.autoreg.sarlstm.config import Config


class Predictor(AutoRegPredictor):
    """Upper class to train and use a LSTM Model"""

    def __init__(self, model_path=None, config=None):
        self.module_class = LightningModule
        self.config_class = Config
        super().__init__(model_path, config, "SARLSTMPredictor")
