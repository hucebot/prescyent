"""methods to deriv a tensor given its feature"""
import torch
from typing import List

from prescyent.dataset.features import Features, Rotation, Any, Coordinate
from prescyent.dataset.features.rotation_methods import (
    get_relative_rotation_from,
    get_absolute_rotation_from,
)
from prescyent.utils.tensor_manipulation import is_tensor_is_batched


def get_relative_tensor_from(
    input_tensor: torch.Tensor,
    basis_tensor: torch.Tensor,
    tensor_features: Features,
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
        input_tensor = torch.unsqueeze(input_tensor, 0)
        basis_tensor = torch.unsqueeze(basis_tensor, 0)
    output = torch.zeros(
        input_tensor.shape, dtype=input_tensor.dtype, device=input_tensor.device
    )
    for feat in tensor_features:
        # Relative for Rotation is the transform rotation
        if isinstance(feat, Rotation):
            output[:, :, :, feat.ids] = get_relative_rotation_from(
                input_tensor[:, :, :, feat.ids], basis_tensor[:, :, :, feat.ids], feat
            )
        # Relative for Any and Coordonate is a substraction
        elif isinstance(feat, Coordinate) or isinstance(feat, Any):
            output[:, :, :, feat.ids] = (
                input_tensor[:, :, :, feat.ids] - basis_tensor[:, :, :, feat.ids]
            )
    if unbatch:
        output = output.squeeze(0)
    return output


def get_absolute_tensor_from(
    input_tensor: torch.Tensor,
    basis_tensor: torch.Tensor,
    tensor_features: Features,
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
        input_tensor = torch.unsqueeze(input_tensor, 0)
        basis_tensor = torch.unsqueeze(basis_tensor, 0)
    output = torch.zeros(
        input_tensor.shape, dtype=input_tensor.dtype, device=input_tensor.device
    )
    for feat in tensor_features:
        # Relative for Rotation is the matmul of rotations matrices
        if isinstance(feat, Rotation):
            output[:, :, :, feat.ids] = get_absolute_rotation_from(
                input_tensor[:, :, :, feat.ids], basis_tensor[:, :, :, feat.ids], feat
            )
        # Relative for Any and Coordonate is a addition
        elif isinstance(feat, Coordinate) or isinstance(feat, Any):
            output[:, :, :, feat.ids] = (
                input_tensor[:, :, :, feat.ids] + basis_tensor[:, :, :, feat.ids]
            )
    if unbatch:
        output = output.squeeze(0)
    return output
