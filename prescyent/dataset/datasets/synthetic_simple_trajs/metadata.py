from prescyent.dataset.features.feature import CoordinateXYZ, RotationEuler

FEATURES = [CoordinateXYZ(range(3)), RotationEuler(range(3, 6))]
POINT_LABELS = ["end_effector"]
POINT_PARENTS = [-1]
