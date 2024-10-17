"simple interface to create a motion dataset from already loaded trajectories"
from prescyent.dataset.dataset import TrajectoriesDataset
from prescyent.dataset.config import TrajectoriesDatasetConfig
from prescyent.dataset.trajectories.trajectories import Trajectories


class CustomDataset(TrajectoriesDataset):
    """Simple class to create a TrajectoriesDataset from existing trajectories"""

    def __init__(
        self,
        config: TrajectoriesDatasetConfig,
        trajectories: Trajectories,
        name: str = "CustomDataset",
    ) -> None:
        self._init_from_config(config, TrajectoriesDatasetConfig)
        self.trajectories = trajectories
        self.DATASET_NAME = name
        super().__init__(name=self.DATASET_NAME)

    def prepare_data(self):
        """no need to prepare data as the trajectories are passed in init"""
        return
