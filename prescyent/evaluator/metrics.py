"""util functions to measure predictors accuracy"""

import torch

from prescyent.utils.tensor_manipulation import is_tensor_is_batched


def get_ade(truth: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    """Average Displacement Error (ADE)
    Average RMSE for all frames
    Lower is better
    """
    return torch.mean(get_rmse(truth, pred))


def get_fde(truth: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    """Final Displacement Error (FDE)
    RMSE at last frame
    Lower is better
    """
    rmse = get_rmse(truth, pred)
    return rmse[-1]


def get_mse(truth: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    """mean squared error (MSE) between ground truth and trajectory prediction"""
    if truth.shape != pred.shape:
        raise AttributeError("Truth and pred tensors must have the same shape")
    if (
        len(pred.shape) > 3
    ):  # when batched, we swap the axes and get MSE for all batche's frame
        truth = torch.transpose(truth, 0, 1)
        pred = torch.transpose(pred, 0, 1)
    output = torch.empty(truth.shape[0])
    for frame, t_frame in enumerate(truth):
        p_frame = pred[frame]
        output[frame] = torch.nn.functional.mse_loss(t_frame, p_frame)
    return output


def get_rmse(truth: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    """root mean squared error (RMSE) between ground truth and trajectory prediction"""
    mse = get_mse(truth, pred)
    return torch.sqrt(mse)


# h36m_time_ms =      [80,    160,  320,   400,   560,   720,   880,  1000]
# h36m_results_keys = ['#2', '#4', '#8', '#10', '#14', '#18', '#22', '#25']
def get_mpjpe(truth: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    if is_tensor_is_batched(truth):
        batch_len = truth.shape[0]
        mpjpe = (
            torch.sum(
                torch.mean(torch.norm(truth * 1000 - pred * 1000, dim=3), dim=2), dim=0
            )
            / batch_len
        )
    else:
        mpjpe = torch.mean(torch.norm(truth * 1000 - pred * 1000, dim=2), dim=1)
    return mpjpe
