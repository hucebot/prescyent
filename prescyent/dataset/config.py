"""Common config elements for motion datasets usage"""
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel

from prescyent.utils.enums import LearningTypes


root_dir = Path(__file__).parent.parent.parent
DEFAULT_DATA_PATH = str(root_dir / "data" / "datasets")


class MotionDatasetConfig(BaseModel):
    """Pydantic Basemodel for MotionDatasets configuration"""

    batch_size: int = 128
    learning_type: LearningTypes = LearningTypes.SEQ2SEQ
    shuffle: bool = True
    num_workers: int = 0
    drop_last: bool = True
    persistent_workers: bool = False
    pin_memory: bool = True
    history_size: int  # number of timesteps as input
    future_size: int  # number of predicted timesteps
    out_dims: Optional[List[int]] = None
    in_dims: Optional[List[int]] = None
