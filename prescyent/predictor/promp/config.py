"""Config elements for Pytorch Lightning Modules usage"""
from pydantic import BaseModel
from prescyent.dataset.config import MotionDatasetConfig


class PrompConfig(BaseModel):
    """Pydantic Basemodel for PrompPredictor configuration"""

    dataset_config: MotionDatasetConfig
    save_path: str = "data/models"
    num_bf: float = 20  # lower = smoother
    ridge_factor: float = 1e-10  # default value
