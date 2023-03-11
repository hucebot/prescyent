"""Config elements for TeleopIcub dataset usage"""
from typing import List, Union
from prescyent.dataset.config import MotionDatasetConfig


class DatasetConfig(MotionDatasetConfig):
    """Pydantic Basemodel for TeleopIcubDataset configuration"""
    url: Union[str, None] = None
    data_path: str = "data/datasets/h36m"
    subsampling_step: int = 2     # subsampling -> 100 Hz to 10Hz
    dimensions: Union[List[int], None] = None  # num features in the data
    used_joints: List[int] = [2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 14, 15, 17, 18, 19,
                             21, 22, 25, 26, 27, 29, 30]  # indexes of the joints
    #type of actions to load
    actions: List[str] =['directions', 'discussion', 'eating', 'greeting',
                         'phoning', 'posing', 'purchases', 'sitting',
                         'sittingdown', 'smoking', 'takingphoto', 'waiting',
                         'walking', 'walkingdog', 'walkingtogether']
    subjects_train: List[str] = ["S1", "S6", "S7", "S8", "S9", "S11"]
    subjects_test: List[str] = ["S5"]
    subjects_val: List[str] = []
