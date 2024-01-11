import numpy as np

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


def _get_metadata():
    """
    code from
    https://github.com/una-dinosauria/human-motion-prediction/blob/master/src/forward_kinematics.py#L100
    dataset metadata + external knowldege needed to build the kinematic tree

    Returns:
      parent: 32-long vector with parent-child relationships in the kinematic tree
      offset: 96-long vector with bone lenghts
      rotInd: 32-long list with indices into angles
      expmapInd: 32-long list with indices into expmap angles
    """

    parent = (
        np.array(
            [
                0,
                1,
                2,
                3,
                4,
                5,
                1,
                7,
                8,
                9,
                10,
                1,
                12,
                13,
                14,
                15,
                13,
                17,
                18,
                19,
                20,
                21,
                20,
                23,
                13,
                25,
                26,
                27,
                28,
                29,
                28,
                31,
            ]
        )
        - 1
    )

    offset = np.array(
        [
            0.000000,
            0.000000,
            0.000000,
            -132.948591,
            0.000000,
            0.000000,
            0.000000,
            -442.894612,
            0.000000,
            0.000000,
            -454.206447,
            0.000000,
            0.000000,
            0.000000,
            162.767078,
            0.000000,
            0.000000,
            74.999437,
            132.948826,
            0.000000,
            0.000000,
            0.000000,
            -442.894413,
            0.000000,
            0.000000,
            -454.206590,
            0.000000,
            0.000000,
            0.000000,
            162.767426,
            0.000000,
            0.000000,
            74.999948,
            0.000000,
            0.100000,
            0.000000,
            0.000000,
            233.383263,
            0.000000,
            0.000000,
            257.077681,
            0.000000,
            0.000000,
            121.134938,
            0.000000,
            0.000000,
            115.002227,
            0.000000,
            0.000000,
            257.077681,
            0.000000,
            0.000000,
            151.034226,
            0.000000,
            0.000000,
            278.882773,
            0.000000,
            0.000000,
            251.733451,
            0.000000,
            0.000000,
            0.000000,
            0.000000,
            0.000000,
            0.000000,
            99.999627,
            0.000000,
            100.000188,
            0.000000,
            0.000000,
            0.000000,
            0.000000,
            0.000000,
            257.077681,
            0.000000,
            0.000000,
            151.031437,
            0.000000,
            0.000000,
            278.892924,
            0.000000,
            0.000000,
            251.728680,
            0.000000,
            0.000000,
            0.000000,
            0.000000,
            0.000000,
            0.000000,
            99.999888,
            0.000000,
            137.499922,
            0.000000,
            0.000000,
            0.000000,
            0.000000,
        ]
    )
    offset = offset.reshape(-1, 3)

    rotInd = [
        [5, 6, 4],
        [8, 9, 7],
        [11, 12, 10],
        [14, 15, 13],
        [17, 18, 16],
        [],
        [20, 21, 19],
        [23, 24, 22],
        [26, 27, 25],
        [29, 30, 28],
        [],
        [32, 33, 31],
        [35, 36, 34],
        [38, 39, 37],
        [41, 42, 40],
        [],
        [44, 45, 43],
        [47, 48, 46],
        [50, 51, 49],
        [53, 54, 52],
        [56, 57, 55],
        [],
        [59, 60, 58],
        [],
        [62, 63, 61],
        [65, 66, 64],
        [68, 69, 67],
        [71, 72, 70],
        [74, 75, 73],
        [],
        [77, 78, 76],
        [],
    ]

    expmapInd = np.split(np.arange(4, 100) - 1, 32)

    return parent, offset, rotInd, expmapInd
