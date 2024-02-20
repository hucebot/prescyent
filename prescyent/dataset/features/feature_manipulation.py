import copy
from typing import Dict, List

import torch

from prescyent.dataset.features.feature import (
    Feature,
    Rotation,
    RotationRotMat,
    RotationQuat,
)
from prescyent.dataset.features.rotation_methods import (
    convert_rotation_tensor_to,
    convert_to_rotmatrix,
    convert_to_quat,
)
from prescyent.utils.tensor_manipulation import is_tensor_is_batched


def get_distance(
    tensor_a: torch.Tensor,
    tensor_feats_a: List[Feature],
    tensor_b: torch.Tensor,
    tensor_feats_b: List[Feature],
    get_mean: bool = False,
) -> Dict[str, torch.Tensor]:
    """returns a feature aware distance between two tensors

    Args:
        tensor_a (torch.Tensor): first tensor
        tensor_feats_a (List[Feature]): first tensor's features
        tensor_b (torch.Tensor): second tensor
        tensor_feats_b (List[Feature]): second tensor's features

    Returns:
        Dict[str, torch.Tensor]: a distance of each feature of the input tensors
    """
    # convert into same feats with a as the priority
    if not is_tensor_is_batched(tensor_a):
        tensor_a = torch.unsqueeze(tensor_a, 0)
    if not is_tensor_is_batched(tensor_b):
        tensor_b = torch.unsqueeze(tensor_b, 0)
    feats = tensor_feats_a
    if not tensor_feats_a == tensor_feats_b:
        if not features_are_convertible_to(tensor_feats_a, tensor_feats_b):
            if not features_are_convertible_to(tensor_feats_b, tensor_feats_a):
                raise AttributeError(
                    f"Cannot compare {tensor_feats_a} with {tensor_feats_b}"
                )
            tensor_b = convert_tensor_features_to(
                tensor_b, tensor_feats_b, tensor_feats_a
            )
        else:
            tensor_a = convert_tensor_features_to(
                tensor_a, tensor_feats_a, tensor_feats_b
            )
            feats = tensor_feats_b
    # get dist depending on feat
    distances = dict()
    for feat in feats:
        distances[feat.name] = cal_distance_for_feat(
            tensor_a[:, :, :, feat.ids], tensor_b[:, :, :, feat.ids], feat
        )
    if get_mean:
        return {key: torch.mean(tensor) for key, tensor in distances.items()}
    return distances


def cal_distance_for_feat(
    tensor_a: torch.Tensor, tensor_b: torch.Tensor, feat: Feature
) -> torch.Tensor:
    if (
        isinstance(feat, Rotation)
        and not isinstance(feat, RotationQuat)
        # and not isinstance(feat, RotationRotMat)
    ):  # convert rotation to quaternion
        tensor_a = convert_to_quat(tensor_a)
        tensor_b = convert_to_quat(tensor_b)
        feat = RotationQuat(range(4))
    return feat.get_distance(tensor_a, tensor_b)


def convert_tensor_features_to(
    tensor: torch.Tensor, tensor_feats: List[Feature], new_tensor_feats: List[Feature]
) -> torch.Tensor:
    """convert a tensor with given features to new tensor according to given features

    Args:
        tensor (torch.Tensor): tensor
        tensor_feats (List[Feature]): actual features of the tensor
        new_tensor_feats (List[Feature]): expected features of the returned tensor

    Raises:
        AttributeError: Can be raised if the conversion is not possible

    Returns:
        torch.Tensor: new tensor matching new features
    """
    if tensor_feats == new_tensor_feats:
        return tensor
    tf = copy.deepcopy(tensor_feats)
    new_tf = copy.deepcopy(new_tensor_feats)
    unbatch = False
    if not is_tensor_is_batched(tensor):
        unbatch = True
        tensor = tensor.unsqueeze(0)
    new_shapes = list(tensor.shape)[:-1] + [sum([len(feat.ids) for feat in new_tf])]
    new_tensor = torch.zeros(new_shapes, dtype=tensor.dtype, device=tensor.device)
    for feat in new_tf:
        equals = [i for i, _feat in enumerate(tf) if feat == _feat]
        if equals:
            new_tensor[:, :, :, feat.ids] = tensor[:, :, :, tf.pop(equals[0]).ids]
            continue
        alike = [i for i, _feat in enumerate(tf) if feat._is_alike(_feat)]
        if alike:
            new_tensor[:, :, :, feat.ids] = tensor[:, :, :, tf.pop(alike[0]).ids]
            continue
        convertible = [i for i, _feat in enumerate(tf) if _feat._is_convertible(feat)]
        if convertible:
            old_tensor = tensor[:, :, :, tf.pop(convertible[0]).ids]
            if isinstance(feat, Rotation):
                old_tensor = convert_rotation_tensor_to(old_tensor, feat)
            new_tensor[:, :, :, feat.ids] = old_tensor[:, :, :, : len(feat.ids)]
            continue
        raise AttributeError(f"Cannot convert feature any of {tf} to match {feat}")
    if unbatch:
        new_tensor = new_tensor.squeeze(0)
    return new_tensor


def features_are_convertible_to(
    features_a: List[Feature], features_b: List[Feature]
) -> bool:
    feats_a = copy.deepcopy(features_a)
    for feat in features_b:
        equals = [i for i, _feat in enumerate(feats_a) if feat == _feat]
        if equals:
            feats_a.pop(equals[0])
            continue
        alike = [i for i, _feat in enumerate(feats_a) if feat._is_alike(_feat)]
        if alike:
            feats_a.pop(alike[0])
            continue
        convertible = [
            i for i, _feat in enumerate(feats_a) if _feat._is_convertible(feat)
        ]
        if convertible:
            feats_a.pop(convertible[0])
            continue
        return False
    return True
