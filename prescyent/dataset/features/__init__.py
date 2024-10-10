"""
We define features as metadatas of our tensors
They are used to perform feature wise operations
Like normalization, loss calculation, input conversion

Here we'll have the definitions of the features,
and methods to convert them, like from rotation matrices to quaternions
"""
import copy
from typing import List, Tuple
import torch

from prescyent.dataset.features.features import Features
from prescyent.dataset.features.feature.feature import Feature
from prescyent.dataset.features.feature.any import Any
from prescyent.dataset.features.feature.coordinate import (
    Coordinate,
    CoordinateX,
    CoordinateXY,
    CoordinateXYZ,
)
from prescyent.dataset.features.feature.rotation import (
    Rotation,
    RotationQuat,
    RotationRotMat,
    RotationEuler,
    RotationRep6D,
)

from prescyent.dataset.features.feature_manipulation import (
    convert_tensor_features_to,
    features_are_convertible_to,
    get_distance,
)
