BASE_FREQUENCY = 50
POINT_LABELS = [
    "hips_0",
    "right_hip_1",
    "right knee_2",
    "right_foot_3",
    "right_foot_4",
    "right_foot_5",
    "left_hip_6",
    "left_knee_7",
    "left_foot_8",
    "left_foot_9",
    "left_foot_10",
    "crotch_11",
    "spine_12",
    "thorax_13",
    "nose_14",
    "head_15",
    "neck_16",
    "left_shoulder_17",
    "left_elbow_18",
    "left_wrist_19",
    "left_wrist_20",
    "left_hand_21",
    "left_hand_22",
    "left_hand_23",
    "neck_24",
    "right_shoulder_25",
    "right_elbow_26",
    "right_wrist_27",
    "right_wrist_28",
    "right_hand_29",
    "right_hand_30",
    "right_hand_31",
]
POINT_PARENTS = [
    -1,
    0,
    1,
    2,
    3,
    4,
    0,
    6,
    7,
    8,
    9,
    0,
    11,
    12,
    13,
    14,
    12,
    16,
    17,
    18,
    19,
    20,
    19,
    22,
    12,
    24,
    25,
    26,
    27,
    28,
    27,
    30,
]
FILE_LABELS = []
for point in POINT_LABELS:
    FILE_LABELS.append(point + "_x")
    FILE_LABELS.append(point + "_y")
    FILE_LABELS.append(point + "_z")
