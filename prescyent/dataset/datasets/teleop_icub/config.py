"""Config elements for TeleopIcub dataset usage"""
import os
from typing import List, Optional

from prescyent.dataset.config import DEFAULT_DATA_PATH, MotionDatasetConfig


class DatasetConfig(MotionDatasetConfig):
    """Pydantic Basemodel for TeleopIcubDataset configuration"""

    url: str = "https://zenodo.org/record/5913573/files/AndyData-lab-prescientTeleopICub.zip?download=1"
    data_path: str = os.path.join(DEFAULT_DATA_PATH, "AndyData-lab-prescientTeleopICub")
    glob_dir: str = "datasetMultipleTasks/*/p*.csv"
    subsampling_step: int = 10  # subsampling -> 100 Hz to 10Hz
    used_joints: Optional[List[int]] = None  # None == all
    ratio_train: float = 0.8
    ratio_test: float = 0.15
    ratio_val: float = 0.05
    history_size: int = 10  # number of timesteps as input, default to 1s at 10Hz
    future_size: int = 10  # number of predicted timesteps, default to 1s at 10Hz
