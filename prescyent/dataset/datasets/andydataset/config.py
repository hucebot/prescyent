"""Config elements for AndyData-lab-onePerson dataset usage"""
import os
from typing import List, Optional

from prescyent.dataset.config import DEFAULT_DATA_PATH, MotionDatasetConfig


class DatasetConfig(MotionDatasetConfig):
    """Pydantic Basemodel for AndyDataset configuration"""

    # url: str = "https://zenodo.org/records/3254403/files/xens_mnvx.zip?download=1"
    data_path: str = os.path.join(
        DEFAULT_DATA_PATH, "AndyData-lab-onePerson", "xsens_mnvx"
    )
    glob_dir: str = "Participant_*.mvnx"
    pt_glob_dir: str = "Participant_*.pt"
    use_pt: bool = True
    subsampling_step: int = 24  # subsampling -> 240 Hz to 10Hz
    used_joints: Optional[List[int]] = None  # None == all
    make_joints_position_relative_to: Optional[
        int
    ] = None  # None == Relative to world, else relative to joint with id == int
    ratio_train: float = 0.8
    ratio_test: float = 0.15
    ratio_val: float = 0.05
    history_size: int = 10  # number of timesteps as input, default to 1s at 10Hz
    future_size: int = 10  # number of predicted timesteps, default to 1s at 10Hz
