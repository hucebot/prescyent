"""constants and default values for the dataset"""

from prescyent.dataset.features import CoordinateXYZ, Features, RotationQuat


BASE_FREQUENCY = 240
POINT_LABELS = [
    "Pelvis",  # 0
    "L5",  # 1
    "L3",  # 2
    "T12",  # 3
    "T8",  # 4
    "Neck",  # 5
    "Head",  # 6
    "RightShoulder",  # 7
    "RightUpperArm",  # 8
    "RightForeArm",  # 9
    "RightHand",  # 10
    "LeftShoulder",  # 11
    "LeftUpperArm",  # 12
    "LeftForeArm",  # 13
    "LeftHand",  # 14
    "RightUpperLeg",  # 15
    "RightLowerLeg",  # 16
    "RightFoot",  # 17
    "RightToe",  # 18
    "LeftUpperLeg",  # 19
    "LeftLowerLeg",  # 20
    "LeftFoot",  # 21
    "LeftToe",  # 22
]
POINT_PARENTS = [
    -1,  # 0: Pelvis
    0,  # 1: L5
    1,  # 2: L3
    2,  # 3: T12
    3,  # 4: T8
    4,  # 5: Neck
    5,  # 6: Head
    4,  # 7: RightShoulder
    7,  # 8: RightUpperArm
    8,  # 9: RightForeArm
    9,  # 10: RightHand
    4,  # 11: LeftShoulder
    11,  # 12: LeftUpperArm
    12,  # 13: LeftForeArm
    13,  # 14: LeftHand
    0,  # 15: RightUpperLeg
    15,  # 16: RightLowerLeg
    16,  # 17: RightFoot
    17,  # 18: RightToe
    0,  # 19: LeftUpperLeg
    19,  # 20: LeftLowerLeg
    20,  # 21: LeftFoot
    21,  # 22: LeftToe
]

SCHEMA = "{http://www.xsens.com/mvn/mvnx}"
DEFAULT_FEATURES = Features([CoordinateXYZ(range(3)), RotationQuat(range(3, 7))])
CONTEXT_KEYS = [
    "velocity",
    "acceleration",
    "angularVelocity",
    "angularAcceleration",
    "sensorFreeAcceleration",
    "sensorMagneticField",
    "sensorOrientation",
    "jointAngle",
    "jointAngleXZY",
    "centerOfMass",
]
