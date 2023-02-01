"""Pytorch module et Lightning module for Seq2Seq"""

from prescyent.predictor.lightning.predictor import LightningPredictor
from prescyent.predictor.lightning.seq2seq.module import Seq2SeqModule
from prescyent.predictor.lightning.seq2seq.config import Seq2SeqConfig


class Seq2SeqPredictor(LightningPredictor):
    """Upper class to train and use a Seq2Seq Model"""

    def __init__(self, model_path=None, config=None):
        self.module_class = Seq2SeqModule
        self.config_class = Seq2SeqConfig
        super().__init__(model_path, config)
