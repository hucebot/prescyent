"""util functions for tensors"""
import torch


def flatten_list_of_preds(preds: torch.Tensor):
    # if we have a list of preds
    if len(preds.shape) == 3:
        # we flatten the prediction to the last output of each prediciton
        # (seq_len, input_size, feature_size) -> (seq_len, feature_size)
        flatten_preds = torch.zeros(preds.shape[0], preds.shape[2])
        for j, pred in enumerate(preds):
            flatten_preds[j] = pred[-1]
        preds = flatten_preds
    return preds
