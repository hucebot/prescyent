"""Pytorch module et Lightning module for Seq2Seq"""

from prescyent.predictor.lightning.models.sequence.predictor import SequencePredictor
from .module import Seq2SeqTorchModule
from .config import Seq2SeqConfig


class Seq2SeqPredictor(SequencePredictor):
    """Upper class to train and use a Seq2Seq Model"""

    PREDICTOR_NAME = "Seq2SeqPredictor"
    """unique name for this predictror"""
    module_class = Seq2SeqTorchModule
    """LightningModule class used in this predictor"""
    config_class = Seq2SeqConfig
    """PredictorConfig class used in this predictor"""

    def __init__(self, config: Seq2SeqConfig, skip_build: bool = False):
        super().__init__(config=config, name=self.PREDICTOR_NAME, skip_build=skip_build)
