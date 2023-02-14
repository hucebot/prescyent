"""Pytorch module et Lightning module for LSTMs"""

from prescyent.predictor.lightning.predictor import LightningPredictor
from prescyent.predictor.lightning.lstm.module import LightningModule
from prescyent.predictor.lightning.lstm.config import Config


class Predictor(LightningPredictor):
    """Upper class to train and use a LSTM Model"""

    def __init__(self, model_path=None, config=None):
        self.module_class = LightningModule
        self.config_class = Config
        super().__init__(model_path, config, "LSTMPredictor")
