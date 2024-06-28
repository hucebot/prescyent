from enum import Enum


class Profilers(str, Enum):
    """Map to a pytorch_lightning.profilers in the lightnin predictor"""

    TORCH = "torch"
    """pytorch_lightning.profilers.PyTorchProfiler"""
    SIMPLE = "simple"
    """pytorch_lightning.profilers.SimpleProfiler"""
    ADVANCED = "advanced"
    """pytorch_lightning.profilers.AdvancedProfiler"""
