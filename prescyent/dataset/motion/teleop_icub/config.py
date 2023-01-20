from typing import List, Union
from prescyent.dataset.motion.config import MotionDatasetConfig


class TeleopIcubDatasetConfig(MotionDatasetConfig):
    url = "https://zenodo.org/record/5913573/files/AndyData-lab-prescientTeleopICub.zip?download=1"
    data_path: str = "data/datasets/AndyData-lab-prescientTeleopICub/datasetMultipleTasks"
    glob_dir: str = 'p*.csv'
    subsampling_step: int = 10     # subsampling -> 100 Hz to 10Hz
    dimensions: Union[List[int], None] = [2]         # dimension in the data
    ratio_train: float = .8
    ratio_test: float = .15
    ratio_val: float = .05
