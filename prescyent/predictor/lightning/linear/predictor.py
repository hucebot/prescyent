"""Pytorch module et Lightning module for Linears"""

from prescyent.predictor.lightning.predictor import LightningPredictor
from prescyent.predictor.lightning.linear.module import LinearModule
from prescyent.predictor.lightning.linear.config import LinearConfig


class LinearPredictor(LightningPredictor):
    """Upper class to train and use a Linear Model"""

    def __init__(self, model_path=None, config=None):
        self.module_class = LinearModule
        self.config_class = LinearConfig
        super().__init__(model_path, config)
