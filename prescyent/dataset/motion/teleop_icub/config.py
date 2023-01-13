from prescyent.dataset.motion.config import MotionDatasetConfig


class TeleopIcubDatasetConfig(MotionDatasetConfig):
    url = "https://zenodo.org/record/5913573/files/AndyData-lab-prescientTeleopICub.zip?download=1"
    data_path: str = "data/datasets/AndyData-lab-prescientTeleopICub/datasetMultipleTasks"
    glob_dir: str = 'p*.csv'
    skip_data: int = 10     # subsampling -> 100 Hz to 10Hz
    column: int = 2         # column in the data
    ratio_train: float = .8
    ratio_test: float = .15
    ratio_val: float = .05
