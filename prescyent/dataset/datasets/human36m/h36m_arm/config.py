from enum import Enum

from prescyent.dataset.datasets.human36m.config import (
    DatasetConfig as H36MDatasetConfig,
)


class Arms(str, Enum):
    LEFT = "left"
    RIGHT = "right"


class DatasetConfig(H36MDatasetConfig):
    """Pydantic Basemodel for Dataset configuration"""

    bimanual: bool = True  # If bimanual, subsample dataset to both arms,
    # else we use the following:
    main_arm: Arms = None  # For mono arm, decide which is main arm
    use_both_arms: bool = False  # Can use the second arm to augment data
    # TODO
    # mirror_second_arm: bool = False   # For data augmentation mirror second arms
