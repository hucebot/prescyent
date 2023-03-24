"""Pytorch module et Lightning module for LSTMs"""

from prescyent.predictor.lightning.predictor import LightningPredictor
from prescyent.predictor.lightning.autoreg.sarlstm.module import TorchModule
from prescyent.predictor.lightning.autoreg.sarlstm.config import Config


class Predictor(LightningPredictor):
    """Upper class to train and use a LSTM Model"""
    PREDICTOR_NAME = "SARLSTMPredictor"

    def __init__(self, model_path=None, config=None):
        self.module_class = TorchModule
        self.config_class = Config
        super().__init__(model_path, config, self.PREDICTOR_NAME)
