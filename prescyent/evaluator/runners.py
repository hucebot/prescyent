"""module to run model and methods evaluation"""
from pathlib import Path
from typing import Callable, List, Tuple
import timeit

import torch
from prescyent.dataset.trajectories import Trajectory
from prescyent.evaluator.eval_result import EvaluationResult, EvaluationSummary

from prescyent.evaluator.metrics import get_ade, get_fde
from prescyent.evaluator.plotting import plot_trajectory_prediction, plot_multiple_predictors
from prescyent.utils.tensor_manipulation import flatten_list_of_preds


def pred_trajectory(trajectory: Trajectory, predictor: Callable,
                    history_size: int = 10, future_size: int = None,
                    eval_on_last_pred: bool = False,
                    skip_partial_input: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """loops a predictor over a whole trajectory

    Args:
        trajectory (torch.Tensor): a tensor of positions to predict
        predictor (Callable): Any predictor module (or any callable)
        history_size (int, optional): sequence size for the predictor input. Defaults to 10.
        eval_on_last_pred (bool, optional): For each prediction loop, set this to
        True if you only want to retrieve the last prediction of the model,
        False if you want the wholde predicted sequence, defaults to True. Defaults to False.
        skip_partial_input (bool, optional): Set this flag to True to skip a generated sample
        that wont be of len === history_size. Defaults to True.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: tuple of generated inputs and predictions
    """
    inputs = torch.Tensor()
    preds = torch.Tensor()
    for i in range(0, len(trajectory), history_size):
        input_sample = trajectory.scaled_tensor[i:i + history_size]
        if skip_partial_input and input_sample.shape[0] != history_size:
            continue
        prediction = predictor(input_sample, future_size=future_size)
        inputs = torch.cat((inputs, input_sample))
        if eval_on_last_pred:
            prediction = torch.unsqueeze(prediction[-1], 0)
        preds = torch.cat((preds, prediction))
    return preds, inputs

# TODO: remove safely
def eval_trajectory(trajectory: Trajectory,
                    predictor: Callable,
                    history_size: int = 10,
                    future_size: int = None,
                    savefig_path: str = "test.png",
                    eval_on_last_pred: bool = False,
                    unscale_function: Callable = None):
    """runs prediction over a whole trajectory, evaluate and plots the results

    Args:
        trajectory (torch.Tensor): input trajectory to evaluate
        predictor (Callable): Any predictor module (or any callable)
        history_size (int, optional): sequence size for the predictor input. Defaults to 10.
        savefig_path (str, optional): path where to save the plot. Defaults to "test.png".
        eval_on_last_pred (bool, optional): For each prediction loop, set this to
        True if you only want to retrieve the last prediction of the model,
        False if you want the wholde predicted sequence, defaults to True. Defaults to False.
        unscale_function (Callable, optional): function to unscale data before ploting
        If None no unscaling will be done. Defaults to None.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: tuple of evaluation metrics ADE and FDE
    """
    preds, inputs = pred_trajectory(trajectory, predictor,
                                    history_size, future_size,
                                    eval_on_last_pred)
    if unscale_function is not None:    # unscale data if provided function
        preds = unscale_function(preds)
        inputs = unscale_function(inputs)
    # if we only want to look at the last predicted point
    truth = inputs[::history_size] if eval_on_last_pred else inputs
    truth = truth[history_size:]
    ade = get_ade(truth, preds[:-history_size])
    fde = get_fde(truth, preds[:-history_size])
    plot_trajectory_prediction(trajectory,
                               preds, history_size,
                               savefig_path)
    return ade, fde


def pred_trajectory_steps_and_keep_last(trajectory: Trajectory,
                                        predictor: Callable,
                                        history_size: int = None,
                                        history_step: int = 1,
                                        future_size: int = 0):
    preds = predictor(trajectory.scaled_tensor,
                      history_size=history_size,
                      history_step=history_step,
                      future_size=future_size)
    return flatten_list_of_preds(preds)


def eval_trajectory_multiple_predictors(trajectory: Trajectory,
                                        predictors: List[Callable],
                                        history_size: int = None,
                                        savefig_path: str = "test.png",
                                        unscale_function: Callable = None
                                        ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Evaluate a list of predictors on the given trajectory

    Args:
        trajectory (torch.Tensor): tensor of a sample use for prediction input
        predictors (List[Callable]): list of predictors
        history_size (int, optional): if provided, will split the trajectory in
                multiple inputs of size = history_size. Defaults to None.
        savefig_path (str, optional): path to the output plot file.
                Defaults to "test.png".
        unscale_function (Callable, optional): if provided is used to unscale
                the input data. Defaults to None.

    Returns:
        Tuple[List[torch.Tensor], List[torch.Tensor]]: the evaluation metrics for each predictor
                ADE and FDE
    """
    predictions = pred_trajectory_multiple_predictors(trajectory, predictors, history_size)
    if unscale_function is not None:
        predictions = [unscale_function(preds) for preds in predictions]
    if history_size is None:
        history_size = len(trajectory)
    truth = trajectory[history_size:]
    ades = [get_ade(truth, preds[:len(truth)]) for preds in predictions]
    fdes = [get_fde(truth, preds[:len(truth)]) for preds in predictions]
    plot_multiple_predictors(trajectory, predictors, predictions, history_size, savefig_path)
    return ades, fdes


def eval_predictors(predictors: List[Callable],
                    trajectories: List[Trajectory],
                    history_size: int,
                    future_size: int,
                    run_method: str = "windowed",
                    unscale_function: Callable = None,
                    do_plotting: bool = True,
                    saveplot_pattern: str = "%d_%s_prediction",
                    saveplot_dir_path: str = str(Path("data") / "eval"),
                    ) -> List[EvaluationSummary]:
    # TODO: reject impossible values for history_size and future_size
    evaluation_results = [EvaluationSummary() for _ in predictors]
    for t, trajectory in enumerate(trajectories):
        predictions = []
        for p, predictor in enumerate(predictors):
            start = timeit.default_timer()
            if run_method == "windowed":
                # check here
                prediction_list = predictor(trajectory.scaled_tensor,
                                       history_size=history_size,
                                       history_step=history_size,
                                       future_size=future_size)
                prediction = torch.cat(prediction_list, dim=0)
            elif run_method == "step_every_timestamp":
                prediction = pred_trajectory_steps_and_keep_last(trajectory.scaled_tensor,
                                                                 predictor,
                                                                 history_size=history_size,
                                                                 history_step=1,
                                                                 future_size=future_size)
            else:
                raise NotImplementedError("'%s' is not a valid run_method" % run_method)
            elapsed = (timeit.default_timer() - start) * 1000.0
            if unscale_function is not None:
                prediction = unscale_function(prediction)
            truth = trajectory.tensor[:-history_size]
            ade = get_ade(truth, prediction[:len(truth)]).item()
            fde = get_fde(truth, prediction[:len(truth)]).item()
            evaluation_results[p].results.append(EvaluationResult(ade, fde, elapsed))

            if do_plotting:
                savefig_path = str(Path(saveplot_dir_path) / (saveplot_pattern % (t, predictor)))
                plot_trajectory_prediction(trajectory, prediction,
                                           step=history_size, savefig_path=savefig_path)
            predictions.append(prediction)
        if do_plotting:
            savefig_path = str(Path(saveplot_dir_path) / (saveplot_pattern % (t, "all")))
            plot_multiple_predictors(trajectory, predictors, predictions,
                                     step=history_size, savefig_path=savefig_path)
    return evaluation_results
