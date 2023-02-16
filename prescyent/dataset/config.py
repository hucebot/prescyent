"""Common config elements for motion datasets usage"""
from enum import Enum

from pydantic import BaseModel


class LearningTypes(Enum):
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
    history_size = 10       # number of timesteps as input
    future_size = 10      # number of predicted timesteps
