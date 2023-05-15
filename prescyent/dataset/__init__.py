"""
Download and/or load dataset
Manipulate and Visualize the dataset
Configure the use of the dataset

Built with torch dataset and dataloader
"""

from prescyent.dataset.human36m import H36MDataset, H36MDatasetConfig
from prescyent.dataset.sine import SineDataset, SineDatasetConfig
from prescyent.dataset.teleop_icub import TeleopIcubDataset, TeleopIcubDatasetConfig


DATASET_LIST = [H36MDataset,
                SineDataset,
                TeleopIcubDataset,
                ]

DATASET_MAP = {p.DATASET_NAME: p for p in DATASET_LIST}
