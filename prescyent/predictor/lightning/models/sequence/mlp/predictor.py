"""Pytorch module et Lightning module for MLP"""

from prescyent.predictor.lightning.models.sequence.predictor import SequencePredictor

from .module import MlpTorchModule
from .config import MlpConfig


class MlpPredictor(SequencePredictor):
    """Upper class to train and use a MLP Model"""

    PREDICTOR_NAME = "MlpPredictor"
    """unique name for this predictror"""
    module_class = MlpTorchModule
    """LightningModule class used in this predictor"""
    config_class = MlpConfig
    """PredictorConfig class used in this predictor"""

    def __init__(self, config: MlpConfig, skip_build: bool = False):
        super().__init__(config=config, name=self.PREDICTOR_NAME, skip_build=skip_build)
