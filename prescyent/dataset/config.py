"""Common config elements for motion datasets usage"""
from pathlib import Path

from pydantic import BaseModel

from prescyent.utils.enums import LearningTypes


root_dir = Path(__file__).parent.parent.parent
DEFAULT_DATA_PATH = str(root_dir / "data" / "datasets")


class MotionDatasetConfig(BaseModel):
    """Pydantic Basemodel for MotionDatasets configuration"""

    batch_size = 128
    learning_type: LearningTypes = LearningTypes.SEQ2SEQ
    shuffle = True
    num_workers = 0
    drop_last: bool = True
    persistent_workers = False
    pin_memory = True
    history_size: int  # number of timesteps as input
    future_size: int  # number of predicted timesteps
