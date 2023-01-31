"""Pytorch module et Lightning module for LSTMs"""

from prescyent.predictor.lightning.predictor import LightningPredictor
from prescyent.predictor.lightning.lstm.module import LSTMModule
from prescyent.predictor.lightning.lstm.config import LSTMConfig


class LSTMPredictor(LightningPredictor):
    """Upper class to train and use a LSTM Model"""

    def __init__(self, model_path=None, config=None):
        self.module_class = LSTMModule
        self.config_class = LSTMConfig
        super().__init__(model_path, config)
