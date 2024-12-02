"""Config elements for Linear Pytorch Lightning module usage"""

from typing import Literal, Optional
from pydantic import Field

from prescyent.predictor.lightning.configs.module_config import ModuleConfig
from prescyent.utils.enums import TrajectoryDimensions


class SiMLPeConfig(ModuleConfig):
    """Pydantic Basemodel for MLP Module configuration"""

    num_layers: int = Field(48, gt=0)
    """Number of MLPBlock"""
    dct: bool = True
    """If True, we apply Discrete Cosine Transform over the input and inverse cosine transform over the output"""
    spatial_fc_only: bool = False
    """If True, MLPBlock will have the Spatial features as inputs, else it's temporals"""
    temporal_fc_in: bool = False
    """If True, First FC Layer will be have the temporal features as inputs, else it's spatial features"""
    temporal_fc_out: bool = False
    """If True, Last FC Layer will be have the temporal features as inputs, else it's spatial features"""
    mpl_blocks_norm: Optional[
        Literal[
            TrajectoryDimensions.BATCH,
            TrajectoryDimensions.ALL,
            TrajectoryDimensions.SPATIAL,
            TrajectoryDimensions.TEMPORAL,
        ]
    ] = TrajectoryDimensions.SPATIAL
    """Normalization used in each MLPBlock"""
