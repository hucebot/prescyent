"""
Download and/or load dataset
Manipulate and Visualize the dataset
Configure the use of the dataset
"""

from prescyent.dataset.datasets.human36m import H36MDataset, H36MDatasetConfig
from prescyent.dataset.datasets.human36m.h36m_arm import (
    H36MArmDataset,
    H36MArmDatasetConfig,
)
from prescyent.dataset.datasets.teleop_icub import (
    TeleopIcubDataset,
    TeleopIcubDatasetConfig,
)
from prescyent.dataset.datasets.synthetic_simple_trajs import (
    SSTDataset,
    SSTDatasetConfig,
)
from prescyent.dataset.datasets.synthetic_circle_clusters import (
    SCCDataset,
    SCCDatasetConfig,
)
from prescyent.dataset.datasets.andydataset import (
    AndyDataset,
    AndyDatasetConfig,
)
from prescyent.dataset.trajectories import Trajectories, Trajectory
from prescyent.dataset.config import TrajectoriesDatasetConfig
from prescyent.dataset.datasets.custom_dataset import CustomDataset


DATASET_LIST = [
    AndyDataset,
    H36MDataset,
    H36MArmDataset,
    SCCDataset,
    SSTDataset,
    TeleopIcubDataset,
]

# Map used for AutoDataset
DATASET_MAP = {p.DATASET_NAME: p for p in DATASET_LIST}
