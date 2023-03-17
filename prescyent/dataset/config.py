"""Common config elements for motion datasets usage"""
from enum import Enum

from pydantic import BaseModel


class LearningTypes(str, Enum):
    SEQ2SEQ = "sequence_2_sequence"
    AUTOREG = "auto_regressive"


class MotionDatasetConfig(BaseModel):
    """Pydantic Basemodel for MotionDatasets configuration"""
    batch_size = 128
    learning_type: LearningTypes = LearningTypes.SEQ2SEQ
    shuffle = True
    num_workers = 0
    persistent_workers = False
    pin_memory = True
    history_size: int       # number of timesteps as input
    future_size: int      # number of predicted timesteps
