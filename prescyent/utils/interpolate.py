"""functions for interpolation"""

from typing import Dict, Iterable, Tuple

import numpy as np
import torch

from prescyent.dataset.features import Features, Rotation, RotationQuat
from prescyent.dataset.features.rotation_methods import convert_rotation_tensor_to


def interpolate_iterable_with_ratio(
    input_list: Iterable, interpolation_ratio: int
) -> Iterable:
    """interpolate iterable with ratio

    Args:
        input_list (Iterable): list to interpolate
        interpolation_ratio (int): ratio of interpolation
    Returns:
        Iterable: output has size (len(input_list) - 1) * ratio + 1
    """

    output_list = []
    for i, _ in enumerate(input_list[:-1]):
        x_interpolated = np.linspace(
            input_list[i], input_list[i + 1], interpolation_ratio + 1
        )
        output_list += list(x_interpolated)[:-1]
    output_list.append(input_list[-1])
    return output_list


def interpolate_trajectory_tensor_with_ratio(
    input_tensor: torch.Tensor, interpolation_ratio: int
) -> torch.Tensor:
    """interpolate trajectory tensor with ratio

    Args:
        input_tensor (torch.Tensor): traj tensor of shape [S, P, D]
        interpolation_ratio (int): ratio of interpolation

    Returns:
        torch.Tensor: interpolated tensor
    """

    assert len(input_tensor.shape) == 3
    input_tensor = torch.transpose(input_tensor, 0, 1)
    input_tensor = torch.transpose(input_tensor, 1, 2)
    # for each dim and for each point we interpolate on sequence
    output = []
    for point_t in input_tensor:
        point = []
        for dim in point_t:
            point.append(interpolate_iterable_with_ratio(dim, interpolation_ratio))
        output.append(point)
    output_tensor = torch.FloatTensor(output)
    output_tensor = torch.transpose(output_tensor, 1, 2)
    output_tensor = torch.transpose(output_tensor, 0, 1)
    return output_tensor


def downsample_trajectory_tensor(
    input_tensor: torch.Tensor, frequency: int, target_freq: int
) -> torch.Tensor:
    """
    Downsamples the input tensor from frequency to target_freq.

    Args:
        input_tensor (torch.Tensor): Input tensor with shape [S, P, D]
        frequency (int): Original frequency of the tensor
        target_freq (int): Desired frequency of the tensor

    Returns:
        torch.Tensor: Downsampled tensor
    """
    assert len(input_tensor.shape) == 3  # works with unbatched tensors
    seq_len = input_tensor.shape[0]
    new_len = int(seq_len / frequency * target_freq)
    indices = torch.linspace(0, seq_len - 1, new_len).long()
    out_tensor = input_tensor[indices]
    return out_tensor


def upsample_trajectory_tensor(
    input_tensor: torch.Tensor,
    tensor_features: Features,
    frequency: int,
    target_freq: int,
) -> torch.Tensor:
    """
    Manually upsamples the input tensor from original_rate to new_rate.

    Args:
        input_tensor (torch.Tensor): Input tensor with shape [T, H, W]
        frequency (int): Original sampling rate of the tensor
        target_freq (int): Desired sampling rate of the tensor

    Returns:
        torch.Tensor: Upsampled tensor
    """
    assert len(input_tensor.shape) == 3  # works with unbatched tensors
    seq_len, num_points, num_dims = input_tensor.shape
    new_len = int(seq_len / frequency * target_freq)
    new_indices = torch.linspace(0, seq_len - 1, new_len)

    output_tensor = torch.zeros(
        (new_len, num_points, num_dims), dtype=input_tensor.dtype
    )

    for i in range(new_len):
        for feat in tensor_features:
            left_index = int(new_indices[i].floor().item())
            right_index = min(left_index + 1, seq_len - 1)
            alpha = new_indices[i] - left_index
            if isinstance(feat, Rotation):
                q1 = input_tensor[left_index, :, feat.ids]
                q2 = input_tensor[right_index, :, feat.ids]
                if not isinstance(feat, RotationQuat):
                    q1 = convert_rotation_tensor_to(q1, RotationQuat(range(4)))
                    q2 = convert_rotation_tensor_to(q2, RotationQuat(range(4)))
                qx = slerp(q1, q2, alpha)
                if not isinstance(feat, RotationQuat):
                    qx = convert_rotation_tensor_to(qx, feat)
                output_tensor[i, :, feat.ids] = qx
            else:
                output_tensor[i, :, feat.ids] = (1 - alpha) * input_tensor[
                    left_index, :, feat.ids
                ] + alpha * input_tensor[right_index, :, feat.ids]
    return output_tensor


def update_tensor_frequency(
    tensor: torch.Tensor,
    freq: float,
    target_freq: float,
    tensor_features: Features = None,
    context: Dict[str, torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """tensor of given frequency is returned to a updated freq

    Args:
        tensor (torch.Tensor): traj tensor of shape [S, P, D]
        freq (float): frequency of the tensor
        target_freq (float): target frequency
        tensor_features (Features, optional): Features of the tensor. Defaults to None.
        context (Dict[str, torch.Tensor], optional): context alongside the tensor. Defaults to None.

    Raises:
        AttributeError: if features are not provided and we want to upsample
        AttributeError: if you try to upsample some context

    Returns:
        Tuple[torch.Tensor, Dict[str, torch.Tensor]: new tensor and context
    """

    assert len(tensor.shape) == 3  # works with unbatched tensors
    if context is None:
        context = {}
    if target_freq == freq:
        return tensor, context
    if target_freq < freq:
        tensor = downsample_trajectory_tensor(tensor, freq, target_freq)
        if context:
            for c_name, c_tensor in context.items():
                context[c_name] = downsample_trajectory_tensor(
                    c_tensor.unsqueeze(1), freq, target_freq
                ).squeeze(1)
    else:
        if context:
            raise AttributeError(
                "We cannot automatically upsample unkown data in context. Please upsample your trajectory and context tensors yourself before initialising the Trajectory object"
            )
        if tensor_features is None:
            raise AttributeError(
                "We cannot upsample tensor with unknown features, please provide them as attributes"
            )
        tensor = upsample_trajectory_tensor(tensor, tensor_features, freq, target_freq)
    return tensor, context


def slerp(q1, q2, t):
    """
    Perform spherical linear interpolation (slerp) between two quaternions q1 and q2.
    from https://www.euclideanspace.com/maths/algebra/realNormedAlgebra/quaternions/slerp/

    Args:
        q1 (torch.Tensor): Starting quaternion
        q2 (torch.Tensor): Ending quaternion
        t (float): Interpolation parameter between 0 and 1

    Returns:
        torch.Tensor: Interpolated quaternion.
    """
    # Ensure the quaternions are normalized
    q1 = q1 / torch.norm(q1)
    q2 = q2 / torch.norm(q2)
    # Compute the cosine of the angle between the two vectors
    dot = (q1 * q2).sum(dim=-1)
    # Reverse the quaternion if needed to get the smallest rotation
    reverse_indexes = dot < 0
    q2[reverse_indexes] = -q2[reverse_indexes]
    dot[reverse_indexes] = -dot[reverse_indexes]
    # Similarity treshold
    DOT_THRESHOLD = 0.9995
    treshold_indices = dot > DOT_THRESHOLD
    dot = torch.clamp(dot, -1.0, 1.0)
    # theta_0 is the angle between input vectors
    theta_0 = torch.acos(dot)
    # theta is the angle between q1 and the result
    theta = theta_0 * t
    v2 = q2 - q1 * dot[:, None]
    v2 = v2 / torch.norm(v2)
    result = q1 * torch.cos(theta[:, None]) + v2 * torch.sin(theta[:, None])
    # If the inputs are too close we linearly interpolate and normalize the result
    result[treshold_indices] = q1[treshold_indices] + t * (
        q2[treshold_indices] - q1[treshold_indices]
    )
    result = result / torch.norm(result)
    # We always get back to the version of the quaternion with a positive w to avoid double cover
    reverse_indices = result[..., -1] < 0
    result[reverse_indices] = -result[reverse_indices]
    return result
