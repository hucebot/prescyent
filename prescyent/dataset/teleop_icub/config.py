"""Config elements for TeleopIcub dataset usage"""
import os
from typing import List, Union

from prescyent.dataset.config import MotionDatasetConfig, DEFAULT_DATA_PATH


class DatasetConfig(MotionDatasetConfig):
    """Pydantic Basemodel for TeleopIcubDataset configuration"""

    url = "https://zenodo.org/record/5913573/files/AndyData-lab-prescientTeleopICub.zip?download=1"
    data_path: str = os.path.join(DEFAULT_DATA_PATH,
                                  "AndyData-lab-prescientTeleopICub",
                                  "datasetMultipleTasks")
    glob_dir: str = "p*.csv"
    subsampling_step: int = 10  # subsampling -> 100 Hz to 10Hz
    used_joints: Union[List[int], None] = None  # None == all
    ratio_train: float = 0.8
    ratio_test: float = 0.15
    ratio_val: float = 0.05
    history_size: int = 10  # number of timesteps as input
    future_size: int = 10  # number of predicted timesteps
