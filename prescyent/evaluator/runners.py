"""module to run model and methods evaluation"""
from typing import Callable, List, Tuple

import torch
from prescyent.dataset.motion.trajectories import Trajectory

from prescyent.evaluator.metrics import get_ade, get_fde
from prescyent.evaluator.plotting import plot_trajectory_prediction, plot_multiple_predictors
from prescyent.utils.tensor_manipulation import flatten_list_of_preds


def pred_trajectory(trajectory: Trajectory, predictor: Callable,
                 input_size: int = 10, eval_on_last_pred: bool = False,
                 skip_partial_input: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """loops a predictor over a whole trajectory

    Args:
        trajectory (torch.Tensor): a tensor of positions to predict
        predictor (Callable): Any predictor module (or any callable)
        input_size (int, optional): sequence size for the predictor input. Defaults to 10.
        eval_on_last_pred (bool, optional): For each prediction loop, set this to
        True if you only want to retrieve the last prediction of the model,
        False if you want the wholde predicted sequence, defaults to True. Defaults to False.
        skip_partial_input (bool, optional): Set this flag to True to skip a generated sample
        that wont be of len === input_size. Defaults to True.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: tuple of generated inputs and predictions
    """
    inputs = torch.Tensor()
    preds = torch.Tensor()
    for i in range(0, len(trajectory), input_size):
        input_sample = trajectory.scaled_tensor[i:i + input_size]
        if skip_partial_input and input_sample.shape[0] != input_size:
            continue
        prediction = predictor(input_sample)
        inputs = torch.cat((inputs, input_sample))
        if eval_on_last_pred:
            prediction = torch.unsqueeze(prediction[-1], 0)
        preds = torch.cat((preds, prediction))
    return preds, inputs


def eval_trajectory(trajectory: Trajectory,
                 predictor: Callable,
                 input_size: int = 10,
                 savefig_path: str = "test.png",
                 eval_on_last_pred: bool = False,
                 unscale_function: Callable = None):
    """runs prediction over a whole trajectory, evaluate and plots the results

    Args:
        trajectory (torch.Tensor): input trajectory to evaluate
        predictor (Callable): Any predictor module (or any callable)
        input_size (int, optional): sequence size for the predictor input. Defaults to 10.
        savefig_path (str, optional): path where to save the plot. Defaults to "test.png".
        eval_on_last_pred (bool, optional): For each prediction loop, set this to
        True if you only want to retrieve the last prediction of the model,
        False if you want the wholde predicted sequence, defaults to True. Defaults to False.
        unscale_function (Callable, optional): function to unscale data before ploting
        If None no unscaling will be done. Defaults to None.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: tuple of evaluation metrics ADE and FDE
    """
    preds, inputs = pred_trajectory(trajectory, predictor, input_size, eval_on_last_pred)
    if unscale_function is not None:    # unscale data if provided function
        preds = unscale_function(preds)
        inputs = unscale_function(inputs)
    # if we only want to look at the last predicted point
    truth = inputs[::input_size] if eval_on_last_pred else inputs
    truth = truth[input_size:]
    ade = get_ade(truth, preds[:-input_size])
    fde = get_fde(truth, preds[:-input_size])
    plot_trajectory_prediction(trajectory, inputs, preds, input_size, savefig_path, eval_on_last_pred)
    return ade, fde


# -- TODO: think of some "prediction_modes" to choose how to iterate over epîsodes
# and unify the behaviors of the eval_trajectory and eval_trajectory_multiple_predictors
def pred_trajectory_multiple_predictors(trajectory: Trajectory,
                                     predictors: List[Callable],
                                     input_size: int = None):
    predictions = []
    for predictor in predictors:
        preds = predictor(trajectory.scaled_tensor, input_size=input_size)
        preds = flatten_list_of_preds(preds)
        predictions.append(preds)
    return predictions


def eval_trajectory_multiple_predictors(trajectory: Trajectory,
                                     predictors: List[Callable],
                                     input_size: int = None,
                                     savefig_path: str = "test.png",
                                     unscale_function: Callable = None
                                     ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Evaluate a list of predictors on the given trajectory

    Args:
        trajectory (torch.Tensor): tensor of a sample use for prediction input
        predictors (List[Callable]): list of predictors
        input_size (int, optional): if provided, will split the trajectory in
                multiple inputs of size = input_size. Defaults to None.
        savefig_path (str, optional): path to the output plot file.
                Defaults to "test.png".
        unscale_function (Callable, optional): if provided is used to unscale
                the input data. Defaults to None.

    Returns:
        Tuple[List[torch.Tensor], List[torch.Tensor]]: the evaluation metrics for each predictor
                ADE and FDE
    """
    predictions = pred_trajectory_multiple_predictors(trajectory, predictors, input_size)
    if unscale_function is not None:
        predictions = [unscale_function(preds) for preds in predictions]
    if input_size is None:
        input_size = len(trajectory)
    truth = trajectory[input_size:]
    ades = [get_ade(truth, preds[:len(truth)]) for preds in predictions]
    fdes = [get_fde(truth, preds[:len(truth)]) for preds in predictions]
    plot_multiple_predictors(trajectory, predictors, predictions, input_size, savefig_path)
    return ades, fdes
