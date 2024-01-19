"""util functions for tensors"""
from typing import Iterable, List, Tuple, Union
import torch


def cat_tensor_with_seq_idx(
    preds: Union[List[torch.Tensor], torch.Tensor], flatt_idx: int = -1
):
    if isinstance(preds, list):
        preds = torch.stack(preds, dim=0)
    # if we have a list of preds
    if len(preds.shape) == 4:
        # we flatten the prediction to the last output of each prediciton
        # (seq_len, input_size, num_points, num_dim) -> (seq_len, num_points, num_dim)
        cat_preds = torch.zeros(preds.shape[0], preds.shape[2], preds.shape[3])
        for j, pred in enumerate(preds):
            cat_preds[j] = pred[flatt_idx]
        preds = cat_preds
    return preds


def cat_list_with_seq_idx(preds: torch.Tensor, flatt_idx: int = -1):
    # we flatten the prediction to the last output of each prediciton
    # list[Tensor(future_size, feature_size)] of len == pred_len
    #   -> Tensor(pred_len, feature_size)
    return torch.cat(
        [preds[0][:flatt_idx]] + [pred[flatt_idx].unsqueeze(0) for pred in preds], dim=0
    )


def is_tensor_is_batched(iterable: Iterable):
    return isinstance(iterable, torch.Tensor) and len(iterable.shape) >= 4


def trajectory_tensor_get_dim_limits(
    tensor: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """get min max values for each dim of a trajectory shaped tensor
    for example with a tensor of shape (100, 5, 3),
    where dims are x,y,z, we get the following result:
        ([min(x), min(y), min(z)], [max(x), max(y), max(z)])
    """
    tensor = tensor.transpose(1, 2)
    min_t = torch.min(tensor, dim=2)
    min_t = min_t.values.transpose(0, 1)
    min_t = torch.min(min_t, dim=1)
    max_t = torch.max(tensor, dim=2)
    max_t = max_t.values.transpose(0, 1)
    max_t = torch.max(max_t, dim=1)
    return min_t.values, max_t.values

