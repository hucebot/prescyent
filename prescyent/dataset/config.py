"""Common config elements for motion datasets usage"""
from pydantic import BaseModel

from prescyent.utils.enums import LearningTypes


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
