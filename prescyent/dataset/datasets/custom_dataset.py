"simple interface to create a motion dataset from already loaded trajectories"
from prescyent.dataset.dataset import MotionDataset
from prescyent.dataset.config import MotionDatasetConfig
from prescyent.dataset.trajectories.trajectories import Trajectories


class CustomDataset(MotionDataset):
    """Simple class to create a MotionDataset from existing trajectories"""

    def __init__(
        self,
        config: MotionDatasetConfig,
        trajectories: Trajectories,
        name: str = "CustomDataset",
        load_data_at_init: bool = True,
    ) -> None:
        self._init_from_config(config, MotionDatasetConfig)
        self.trajectories = trajectories
        self.DATASET_NAME = name
        super().__init__(name=self.DATASET_NAME, load_data_at_init=load_data_at_init)

    def prepare_data(self):
        """no need to prepare data as the trajectories are passed in init"""
        return
