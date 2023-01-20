from prescyent.dataset.base_config import DatasetConfig


class MotionDatasetConfig(DatasetConfig):
    input_size = 10       # number of timesteps as input
    output_size = 10      # number of predicted timesteps
    # TODO see torch Dataset args for more
