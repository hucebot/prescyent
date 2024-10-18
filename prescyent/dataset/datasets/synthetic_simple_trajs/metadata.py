"""constants and default values for the dataset"""

from prescyent.dataset.features import CoordinateXYZ, Features, RotationEuler

DEFAULT_FEATURES = Features([CoordinateXYZ(range(3)), RotationEuler(range(3, 6))])
POINT_LABELS = ["end_effector"]
POINT_PARENTS = [-1]
