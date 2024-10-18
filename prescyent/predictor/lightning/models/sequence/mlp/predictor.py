"""Pytorch module et Lightning module for MLP"""

from prescyent.predictor.lightning.models.sequence.predictor import SequencePredictor
from prescyent.predictor.lightning.models.sequence.mlp.module import TorchModule
from prescyent.predictor.lightning.models.sequence.mlp.config import Config


class Predictor(SequencePredictor):
    """Upper class to train and use a MLP Model"""

    PREDICTOR_NAME = "MlpPredictor"
    """unique name for this predictror"""
    module_class = TorchModule
    """LightningModule class used in this predictor"""
    config_class = Config
    """PredictorConfig class used in this predictor"""

    def __init__(self, config: Config, skip_build: bool = False):
        super().__init__(config=config, name=self.PREDICTOR_NAME, skip_build=skip_build)
