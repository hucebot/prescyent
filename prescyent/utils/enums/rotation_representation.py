from enum import Enum


class RotationRepresentation(str, Enum):
    QUATERNIONS = "quaternions"
    ROTMATRICES = "rotmatrices"
    ROTVECTORS = "rotvectors"
    RODRIGUES = "rodrigues"
    EULER = "euler"
    INRIA_WBC = "inria_wbc"  # TODO REMOVE: temporary for data remaping
