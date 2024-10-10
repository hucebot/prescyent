"""Config elements for Normalizing usage"""
from typing import Optional

from prescyent.base_config import BaseConfig
from prescyent.dataset.config import MotionDatasetConfig
from prescyent.scaler.config import ScalerConfig
from prescyent.utils.enums import LearningTypes


class PredictorConfig(BaseConfig):
    """Pydantic Basemodel for Normalizing configuration"""

    name: Optional[str] = None
    """The name of the predictor.
    If None, default to the class value.
    WARNING, If you override default value, AutoPredictor won't be able to load your Predictor"""
    version: Optional[int] = None
    """A version number for this instance of the predictor.
    If None, we'll use TensorBoardLogger logic to aquire a version number from the log path"""
    save_path: str = "data/models"
    """Directory where the model will log and save"""
    dataset_config: MotionDatasetConfig
    """The MotionDatasetConfig used to understand the dataset and its tensor"""
    scaler_config: Optional[ScalerConfig] = None
    """The ScalerConfig used instanciate the scaler of this predictor.
    If None, we'll not use a scaler ahead of the predictor"""

    @property
    def in_sequence_size(self):
        return self.dataset_config.history_size

    @property
    def out_sequence_size(self):
        if self.dataset_config.learning_type == LearningTypes.SEQ2ONE:
            # If we predict one value instead of a sequence
            return 1
        return self.dataset_config.future_size

    @property
    def in_points_dims(self):
        return self.dataset_config.num_in_dims * self.dataset_config.num_in_points

    @property
    def out_points_dims(self):
        return self.dataset_config.num_out_dims * self.dataset_config.num_out_points
