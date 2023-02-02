"""module for model and methods evaluation"""
from typing import Callable, Tuple
import torch
import matplotlib.pyplot as plt

from prescyent.evaluator.metrics import get_ade, get_fde
from prescyent.evaluator.plotting import plot_episode_prediction


def pred_episode(episode: torch.Tensor, predictor: Callable,
                 step: int=10, eval_on_last_pred: bool=False,
                 skip_partial_input=True) -> Tuple[torch.Tensor, torch.Tensor]:
    """loops a predictor over a whole episode

    Args:
        episode (torch.Tensor): a tensor of positions to predict
        predictor (Callable): Any predictor module (or any callable)
        step (int, optional): sequence size for the predictor input. Defaults to 10.
        eval_on_last_pred (bool, optional): For each prediction loop, set this to
        True if you only want to retrieve the last prediction of the model,
        False if you want the wholde predicted sequence, defaults to True. Defaults to False.
        skip_partial_input (bool, optional): Set this flag to True to skip a generated sample
        that wont be of len === step. Defaults to True.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: tuple of generated inputs and predictions
    """
    inputs = torch.Tensor()
    preds = torch.Tensor()
    for i in range(0, len(episode), step):
        input_sample = episode[i:i + step]
        if skip_partial_input and input_sample.shape[0] != step:
            continue
        prediction = predictor(input_sample)
        inputs = torch.cat((inputs, input_sample))
        if eval_on_last_pred:
            prediction = torch.unsqueeze(prediction[-1], 0)
        preds = torch.cat((preds, prediction))
    return preds, inputs


def eval_episode(episode: torch.Tensor,
                 predictor: Callable,
                 step: int=10,
                 savefig_path: str="test.png",
                 eval_on_last_pred: bool=True,
                 unscale_function: Callable=None):
    """runs prediction over a whole episode, evaluate and plots the results

    Args:
        episode (torch.Tensor): input episode to evaluate
        predictor (Callable): Any predictor module (or any callable)
        step (int, optional): sequence size for the predictor input. Defaults to 10.
        savefig_path (str, optional): path where to save the plot. Defaults to "test.png".
        eval_on_last_pred (bool, optional): For each prediction loop, set this to
        True if you only want to retrieve the last prediction of the model,
        False if you want the wholde predicted sequence, defaults to True. Defaults to False.
        unscale_function (Callable, optional): function to unscale data before ploting
        If None no unscaling will be done. Defaults to None.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: tuple of evaluation metrics ADE and FDE
    """
    preds, inputs = pred_episode(episode, predictor, step, eval_on_last_pred)
    if unscale_function is not None:    # unscale data if provided function
        preds = unscale_function(preds)
        inputs = unscale_function(inputs)
    # if we only want to look at the last predicted point
    truth = inputs[::step] if eval_on_last_pred else inputs
    truth = truth[step:]
    ade = get_ade(truth, preds[:-step])
    fde = get_fde(truth, preds[:-step])
    plot_episode_prediction(inputs, preds, step, savefig_path, eval_on_last_pred)
    return ade, fde
