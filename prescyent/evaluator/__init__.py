"""module for model and methods evaluation"""
from typing import Callable
import torch
import matplotlib.pyplot as plt

from prescyent.evaluator.metrics import get_ade, get_fde
from prescyent.evaluator.plotting import plot_episode_prediction


def pred_episode(episode: torch.Tensor, predictor:Callable, step: int=10,
                 eval_on_last_pred: bool=True) -> torch.Tensor:
    """prediction loop over an episode

    :param episode: a tensor of positions to predict
    :type episode: torch.Tensor
    :param predictor: Any predictor module (or any callable)
    :type predictor: Callable
    :param step: sequence size for the predictor input, defaults to 10
    :type step: int, optional
    :param eval_on_last_pred: For each prediction loop, set this to
    True if you only want to retrieve the last prediction of the model,
    False if you want the wholde predicted sequence, defaults to True
    :type eval_on_last_pred: bool, optional
    :return: the models predictions for the episode
    :rtype: torch.Tensor
    """
    inputs = torch.Tensor()
    preds = torch.Tensor()
    # stop = "len(episode) - step" because we dont want to predict future without truth to compare
    for i in range(0, len(episode) - step, step):
        input = episode[i:i + step]
        prediction = predictor(input)
        inputs = torch.cat((inputs, input))
        if eval_on_last_pred:
            prediction = torch.unsqueeze(prediction[-1], 0)
        preds = torch.cat((preds, prediction))
    return preds


def eval_episode(episode, predictor,
                 step: int=10,
                 savefig_path="test.png",
                 eval_on_last_pred=True,
                 unscale_function=None):
    preds = pred_episode(episode, predictor, step, eval_on_last_pred)
    if unscale_function is not None:    # unscale data if provided function
        preds = unscale_function(preds)
        episode = unscale_function(episode)
    if eval_on_last_pred:               # if we only want to look at the last predicted point
        truth = episode[step::step]
    else: # if we want to look at whole predicted sequenc
        truth = episode[step:]
    ade = get_ade(truth, preds)
    fde = get_fde(truth, preds)
    plot_episode_prediction(episode, preds, step, savefig_path, eval_on_last_pred)
    return ade, fde
