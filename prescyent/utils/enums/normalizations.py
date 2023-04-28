from enum import Enum


class Normalizations(str, Enum):
    BATCH_NORM = "batch_norm"
    ALL = "all"
    SPACE_NORM = "space_norm"
    TIME_NORM = "time_norm"
