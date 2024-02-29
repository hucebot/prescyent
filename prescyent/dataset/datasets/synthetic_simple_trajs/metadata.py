from prescyent.dataset.features.feature import CoordinateXYZ, RotationQuat

BASE_FREQUENCY = 10
FEATURES = [CoordinateXYZ(range(3)), RotationQuat(range(3, 7))]
POINT_LABELS = ["end_effector"]
POINT_PARENTS = [-1]
