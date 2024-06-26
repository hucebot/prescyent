"simple interface to create a motion dataset from already loaded trajectories"
from prescyent.dataset.dataset import MotionDataset
from prescyent.dataset.config import MotionDatasetConfig
from prescyent.dataset.trajectories.trajectories import Trajectories
from prescyent.dataset.datasamples import MotionDataSamples


class CustomDataset(MotionDataset):
    def __init__(
        self,
        config,
        trajectories: Trajectories,
        name="CustomDataset",
    ) -> None:
        self._init_from_config(config, MotionDatasetConfig)
        self.trajectories = trajectories
        self.DATASET_NAME = name
        super().__init__(self.DATASET_NAME)
