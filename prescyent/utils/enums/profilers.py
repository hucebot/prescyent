from enum import Enum


class Profilers(str, Enum):
    TORCH = "torch"
    SIMPLE = "simple"
    ADVANCED = "advanced"
