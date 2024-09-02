from enum import Enum


class Scalers(str, Enum):
    """Map to a given scaling method"""

    NORMALIZATION = "normalization"
    STANDARDIZATION = "standardization"
