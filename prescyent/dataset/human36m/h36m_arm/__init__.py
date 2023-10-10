"""
Subset of the h36m dataset using both arms, or one
to remove noise from other movements,
we express the arms' joints positions relatively to the shoulder's joint position
"""

from .dataset import Dataset as H36MArmDataset
from .config import DatasetConfig as H36MArmDatasetConfig
