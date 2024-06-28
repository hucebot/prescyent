"""Config elements for Pytorch Lightning Modules usage"""
from prescyent.dataset.config import MotionDatasetConfig
from prescyent.base_config import BaseConfig


class PrompConfig(BaseConfig):
    """Pydantic Basemodel for PrompPredictor configuration"""

    dataset_config: MotionDatasetConfig
    """The MotionDatasetConfig used to understand the dataset and its tensor"""
    save_path: str = "data/models"
    """Directory where the model will log and save"""
    num_bf: float = 20
    """Lower = smoother"""
    ridge_factor: float = 1e-10
    """Regularization parameter of the ridge regression"""
