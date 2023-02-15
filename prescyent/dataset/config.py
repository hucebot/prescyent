"""Common config elements for motion datasets usage"""
from pydantic import BaseModel


class MotionDatasetConfig(BaseModel):
    """Pydantic Basemodel for MotionDatasets configuration"""
    batch_size = 128
    shuffle = True
    num_workers = 0
    persistent_workers = False
    pin_memory = True
    input_size = 10       # number of timesteps as input
    output_size = 10      # number of predicted timesteps
    # TODO see torch Dataset args for more
