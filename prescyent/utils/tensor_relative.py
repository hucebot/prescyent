import torch
from typing import List

from prescyent.dataset.features import (
    Feature,
    Rotation,
    Any,
    Coordinate,
    get_relative_rotation_from,
    get_absolute_rotation_from,
)
from prescyent.utils.tensor_manipulation import is_tensor_is_batched


def get_relative_tensor_from(
    input_tensor: torch.Tensor,
    basis_tensor: torch.Tensor,
    tensor_features: List[Feature],
) -> torch.Tensor:
    """return a new tensor relative to basis tensor

    Args:
        input_tensor (torch.Tensor): the input tensor to update (B, N, P, D)
        basis_tensor (torch.Tensor): the new basis tensor (B, 1, P, D)

    Returns:
        torch.Tensor: tensor relative to new basis
    """
    unbatch = False
    if not is_tensor_is_batched(input_tensor):
        unbatch = True
        input_tensor = input_tensor.unsqueeze(0)
        basis_tensor = basis_tensor.unsqueeze(0)
    for feat in tensor_features:
        # Relative for Rotation is the transform rotation
        if isinstance(feat, Rotation):
            input_tensor[:, :, :, feat.ids] = get_relative_rotation_from(
                input_tensor[:, :, :, feat.ids], basis_tensor[:, :, :, feat.ids], feat
            )
        # Relative for Any and Coordonate is a substraction
        elif isinstance(feat, Coordinate) or isinstance(feat, Any):
            input_tensor[:, :, :, feat.ids] = (
                input_tensor[:, :, :, feat.ids] - basis_tensor[:, :, :, feat.ids]
            )
    if unbatch:
        input_tensor = input_tensor.squeeze(0)
    return input_tensor


def get_absolute_tensor_from(
    input_tensor: torch.Tensor,
    basis_tensor: torch.Tensor,
    tensor_features: List[Feature],
) -> torch.Tensor:
    """return a new tensor absolute to world based on basis tensor

    Args:
        input_tensor (torch.Tensor): the input tensor to update
        basis_tensor (torch.Tensor): the tensor used as basis

    Returns:
        torch.Tensor: tensor absolute to world
    """
    unbatch = False
    if not is_tensor_is_batched(input_tensor):
        unbatch = True
        input_tensor = input_tensor.unsqueeze(0)
        basis_tensor = basis_tensor.unsqueeze(0)
    for feat in tensor_features:
        # Relative for Rotation is the matmul of rotations matrices
        if isinstance(feat, Rotation):
            input_tensor[:, :, :, feat.ids] = get_absolute_rotation_from(
                input_tensor[:, :, :, feat.ids], basis_tensor[:, :, :, feat.ids], feat
            )
        # Relative for Any and Coordonate is a addition
        elif isinstance(feat, Coordinate) or isinstance(feat, Any):
            input_tensor[:, :, :, feat.ids] = (
                input_tensor[:, :, :, feat.ids] + basis_tensor[:, :, :, feat.ids]
            )
    if unbatch:
        input_tensor = input_tensor.squeeze(0)
    return input_tensor
