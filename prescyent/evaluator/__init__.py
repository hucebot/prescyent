"""module for model and methods evaluation"""
from typing import Callable
import torch
import matplotlib.pyplot as plt

from prescyent.evaluator.metrics import get_ade, get_fde
from prescyent.evaluator.plotting import plot_episode_prediction


def pred_episode(episode: torch.Tensor, predictor:Callable, step: int=10,
                 eval_on_last_pred: bool=True, skip_partial_input=True) -> torch.Tensor:
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
    :param eval_on_last_pred: For each prediction loop, set this to
    True if you only want to retrieve the last prediction of the model,
    False if you want the wholde predicted sequence, defaults to True
    :type eval_on_last_pred: bool, optional
    :return: the models predictions for the episode
    :rtype: torch.Tensor
    """
    inputs = torch.Tensor()
    preds = torch.Tensor()
    for i in range(0, len(episode), step):
        input = episode[i:i + step]
        if skip_partial_input and input.shape[0] != step:
            continue
        prediction = predictor(input)
        inputs = torch.cat((inputs, input))
        if eval_on_last_pred:
            prediction = torch.unsqueeze(prediction[-1], 0)
        preds = torch.cat((preds, prediction))
    return preds, inputs


def eval_episode(episode, predictor,
                 step: int=10,
                 savefig_path="test.png",
                 eval_on_last_pred=True,
                 unscale_function=None):
    preds, inputs = pred_episode(episode, predictor, step, eval_on_last_pred)
    if unscale_function is not None:    # unscale data if provided function
        preds = unscale_function(preds)
        inputs = unscale_function(inputs)
    # if we only want to look at the last predicted point
    truth = inputs[::step]  if eval_on_last_pred else inputs
    truth = truth[step:]
    ade = get_ade(truth, preds[:-step])
    fde = get_fde(truth, preds[:-step])
    plot_episode_prediction(inputs, preds, step, savefig_path, eval_on_last_pred)
    return ade, fde
