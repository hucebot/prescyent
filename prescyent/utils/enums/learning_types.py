from enum import Enum

class LearningTypes(str, Enum):
    SEQ2SEQ = "sequence_2_sequence"
    AUTOREG = "auto_regressive"
