"""constants and default values for the dataset"""

from prescyent.dataset.features import CoordinateXY, Features

DEFAULT_FEATURES = Features([CoordinateXY(range(2))])
DEFAULT_FREQ = 10
POINT_LABELS = ["end_effector"]
POINT_PARENTS = [-1]
