"""util functions for tensors"""
from typing import Iterable
import torch


def cat_tensor_with_seq_idx(preds: torch.Tensor, flatt_idx: int = -1):
    # if we have a list of preds
    if len(preds.shape) == 3:
        # we flatten the prediction to the last output of each prediciton
        # (seq_len, input_size, feature_size) -> (seq_len, feature_size)
        cat_preds = torch.zeros(preds.shape[0], preds.shape[2])
        for j, pred in enumerate(preds):
            cat_preds[j] = pred[flatt_idx]
        preds = cat_preds
    return preds


def cat_list_with_seq_idx(preds: torch.Tensor, flatt_idx: int = -1):
    # we flatten the prediction to the last output of each prediciton
    # list[Tensor(future_size, feature_size)] of len == pred_len
    #   -> Tensor(pred_len, feature_size)
    return torch.cat(
        [preds[0][:flatt_idx]] + [pred[flatt_idx].unsqueeze(0) for pred in preds],
        dim=0
    )


def is_tensor_is_batched(iterable: Iterable):
    return isinstance(iterable, torch.Tensor) and len(iterable.shape) >= 4
