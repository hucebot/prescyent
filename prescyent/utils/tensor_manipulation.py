"""util functions for tensor manipulations"""
import functools
from typing import Iterable, List, Tuple
import torch


def cat_list_with_seq_idx(
    preds: List[torch.Tensor], flatt_idx: int = -1
) -> torch.Tensor:
    """given a list of T traj tensors [(S, P, D)...], returns a tensor of shape [T, P, D], keeping only the frame flatt_idx from all tensors

    Args:
        preds ([List[torch.Tensor]): list of traj tensors
        flatt_idx (int, optional): ids where to flatten the tensors on. Defaults to -1.

    Returns:
        torch.Tensor: cat of the tensor list on given frame
    """

    if is_tensor_is_batched(preds[0]):
        return torch.cat([pred[:, flatt_idx].unsqueeze(0) for pred in preds], dim=0)
    if is_tensor_is_unbatched(preds[0]):
        return torch.cat([pred[flatt_idx].unsqueeze(0) for pred in preds], dim=0)
    else:
        raise AttributeError(
            "tensors in list have different shapes of trajectory tensors"
        )


def is_tensor_is_batched(iterable: Iterable) -> bool:
    """returns true if iterable is a tensor with 4 dims"""

    return isinstance(iterable, torch.Tensor) and len(iterable.shape) == 4


def is_tensor_is_unbatched(iterable: Iterable) -> bool:
    """returns true if iterable is a tensor with 3 dims"""

    return isinstance(iterable, torch.Tensor) and len(iterable.shape) == 3


def self_auto_batch(function):
    """decorator for seemless batched/unbatched forward methods (or any class method)"""

    @functools.wraps(function)
    def reshape(self, input_tensor, *args, **kwargs):
        """reshape input_tensor and context to have 4 dims"""

        unbatched = is_tensor_is_unbatched(input_tensor)
        if unbatched:
            input_tensor = torch.unsqueeze(input_tensor, dim=0)
            if kwargs.get("context", {}):
                kwargs["context"] = {
                    c_name: c_tensor.unsqueeze(0)
                    for c_name, c_tensor in kwargs["context"].items()
                }
        predictions = function(self, input_tensor, *args, **kwargs)
        if unbatched:
            predictions = torch.squeeze(predictions, dim=0)
        return predictions

    return reshape


def auto_batch(function):
    """decorator for seemless batched/unbatched methods (without self attribute)"""

    @functools.wraps(function)
    def reshape(input_tensor, *args, **kwargs):
        """reshape input_tensor and context to have 4 dims"""

        unbatched = is_tensor_is_unbatched(input_tensor)
        if unbatched:
            input_tensor = torch.unsqueeze(input_tensor, dim=0)
            if kwargs.get("context", None):
                kwargs["context"] = {
                    c_name: c_tensor.unsqueeze(0)
                    for c_name, c_tensor in kwargs["context"].items()
                }
        predictions = function(input_tensor, *args, **kwargs)
        if unbatched:
            predictions = torch.squeeze(predictions, dim=0)
        return predictions

    return reshape


def trajectory_tensor_get_dim_limits(
    tensor: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """get min max values for each dim of a trajectory shaped tensor
    for example with a tensor of shape (100, 5, 3),
    where dims are x,y,z, we get the following result:
        ([min(x), min(y), min(z)], [max(x), max(y), max(z)])

    Args:
        tensor (torch.Tensor): traj tensor

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: ([min(x), min(y), min(z)], [max(x), max(y), max(z)])
    """

    tensor = tensor.transpose(1, 2)
    min_t = torch.min(tensor, dim=2)
    min_t = min_t.values.transpose(0, 1)
    min_t = torch.min(min_t, dim=1)
    max_t = torch.max(tensor, dim=2)
    max_t = max_t.values.transpose(0, 1)
    max_t = torch.max(max_t, dim=1)
    return min_t.values, max_t.values
