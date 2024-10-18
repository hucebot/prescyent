"""constants and default values for the dataset"""

from prescyent.dataset.features import CoordinateXYZ, Features

BASE_FREQUENCY = 100
POINT_LABELS = ["waist", "right_hand", "left_hand"]
POINT_PARENTS = [-1, 0, 0]
DEFAULT_FEATURES = Features([CoordinateXYZ(range(3))])
CONTEXT_KEYS = ["center_of_mass", "icub_dof"]
