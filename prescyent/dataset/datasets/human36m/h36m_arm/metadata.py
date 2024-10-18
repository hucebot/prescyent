"""constants and default values for the dataset"""


LEFT_ARM_LABELS = [
    "left_shoulder_17",
    "left_elbow_18",
    "left_wrist_19",
    "left_wrist_20",
    "left_hand_21",
    "left_hand_22",
    "left_hand_23",
]
RIGHT_ARM_LABELS = [
    "right_shoulder_25",
    "right_elbow_26",
    "right_wrist_27",
    "right_wrist_28",
    "right_hand_29",
    "right_hand_30",
    "right_hand_31",
]

RELATIVE_LEFT_ARM_LABEL = LEFT_ARM_LABELS[0]
RELATIVE_RIGHT_ARM_LABEL = RIGHT_ARM_LABELS[0]
RELATIVE_BOTH_ARMS_LABEL = "hips_0"

MIRROR_AXIS_LABELS = [
    "hips_0",
    "crotch_11",
    "spine_12",
    "thorax_13",
    "neck_16",
    "neck_24",
]

ARM_MAP = {
    "left": (LEFT_ARM_LABELS, RELATIVE_LEFT_ARM_LABEL),
    "right": (RIGHT_ARM_LABELS, RELATIVE_RIGHT_ARM_LABEL),
}
