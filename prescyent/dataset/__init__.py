"""
Download and/or load dataset
Manipulate and Visualize the dataset
Configure the use of the dataset

Built with torch dataset and dataloader
"""

from prescyent.dataset.config import LearningTypes

from prescyent.dataset.human36m import H36MDataset, H36MDatasetConfig
from prescyent.dataset.sine import SineDataset, SineDatasetConfig
from prescyent.dataset.teleop_icub import TeleopIcubDataset, TeleopIcubDatasetConfig
