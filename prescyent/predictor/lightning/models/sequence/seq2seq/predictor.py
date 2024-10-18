"""Pytorch module et Lightning module for Seq2Seq"""

from prescyent.predictor.lightning.models.sequence.predictor import SequencePredictor
from prescyent.predictor.lightning.models.sequence.seq2seq.module import TorchModule
from prescyent.predictor.lightning.models.sequence.seq2seq.config import Config


class Predictor(SequencePredictor):
    """Upper class to train and use a Seq2Seq Model"""

    PREDICTOR_NAME = "Seq2SeqPredictor"
    """unique name for this predictror"""
    module_class = TorchModule
    """LightningModule class used in this predictor"""
    config_class = Config
    """PredictorConfig class used in this predictor"""

    def __init__(self, config: Config, skip_build: bool = False):
        super().__init__(config=config, name=self.PREDICTOR_NAME, skip_build=skip_build)
