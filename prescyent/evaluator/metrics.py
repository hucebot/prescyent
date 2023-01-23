"""util functions to measure predictors accuracy"""

import torch


def get_ade(truth: torch.Tensor, pred: torch.Tensor):
    """Average Displacement Error (ADE)
        Average RMSE for all frames
        Lower is better
    """
    return torch.mean(get_rmse(truth, pred)).item()


def get_fde(truth: torch.Tensor, pred: torch.Tensor):
    """Final Displacement Error (FDE)
        RMSE at last frame
        Lower is better
    """
    rmse = get_rmse(truth, pred)
    return rmse[-1].item()


def get_mse(truth: torch.Tensor, pred: torch.Tensor):
    """mean squared error (MSE) between ground truth and trajectory prediction"""
    if truth.shape != pred.shape:
        raise AttributeError("Truth and pred tensors must have the same shape")
    if len(pred.shape) > 1:     # when batched, we swap the axes and get MSE for all batche's frame
        truth = torch.swapaxes(truth, 0, 1)
        pred = torch.swapaxes(pred, 0, 1)
    output = torch.empty(truth.shape[0])
    for frame, t in enumerate(truth):
        p = pred[frame]
        output[frame] = torch.nn.functional.mse_loss(t, p)
    return output


def get_rmse(truth: torch.Tensor, pred: torch.Tensor):
    """root mean squared error (RMSE) between ground truth and trajectory prediction"""
    mse = get_mse(truth, pred)
    return torch.sqrt(mse)
