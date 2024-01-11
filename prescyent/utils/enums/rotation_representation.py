from enum import Enum


class RotationRepresentation(str, Enum):
    QUATERNIONS = "quaternions"
    ROTMATRICES = "rotmatrices"
    EULER = "euler"
    REP6D = "representation_6d"
    # from Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H. (2020).
    # On the continuity of rotation representations in neural networks.
    # arXiv preprint arXiv:1812.07035.
    RODRIGUES = "rodrigues"
    ROTVECTORS = "rotvectors"
