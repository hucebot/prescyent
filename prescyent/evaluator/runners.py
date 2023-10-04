"""module to run model and methods evaluation"""
from pathlib import Path
from typing import Callable, List, Union
import timeit

import torch
from tqdm import tqdm
from prescyent.dataset.trajectories import Trajectory
from prescyent.evaluator.eval_result import EvaluationResult
from prescyent.evaluator.eval_summary import EvaluationSummary

from prescyent.evaluator.plotting import (
    plot_trajectory_prediction,
    plot_multiple_future,
)
from prescyent.utils.logger import logger, EVAL
from prescyent.utils.tensor_manipulation import cat_list_with_seq_idx


def run_predictor(
    predictor: Callable,
    trajectory: torch.Tensor,
    history_size: int,
    future_size: int,
    run_method: str = "windowed",
    output_all: bool = False,
) -> Union[List[torch.Tensor], torch.Tensor]:
    """loops a predictor over a whole trajectory / tensor

    Args:
        predictor (Callable):  Any predictor module (or any callable)
        trajectory (torch.Tensor): a tensor of positions to predict in the shape
                (batch_len, seq_len, features) or (seq_len, features)
        history_size (int): size used as input for the predictor.
        future_size (int): size used as output for the predictor
        run_method (str, optional): method used to generate the predictions.
                "windowed" will predict with a step == history_size
                        The output is cat as a single Tensor
                "step_every_timestamp" will predict with a step == 1
                Defaults to 'windowed'.
        custom_step (int, optional): step used to loop over the input_tensor.
                If ommited, the step is determmimed by the run_method
                Defaults to None
        output_all (bool, optional): if True, will simply output the list of
                results, otherwise does the run_method postprocessing
                Defaults to False.

    Raises:
        NotImplementedError: if the run_method isn't recognized

    Returns:
        List[torch.Tensor]: the list of predictions
    """
    if run_method == "windowed":
        history_step = future_size
        prediction = predictor(
            trajectory,
            history_size=history_size,
            history_step=history_step,
            future_size=future_size,
        )
        if (
            not output_all
        ):  # here we can produce a continous prediction with a simple cat
            prediction = torch.cat(prediction, dim=0)
    elif run_method == "step_every_timestamp":
        prediction = predictor(
            trajectory,
            history_size=history_size,
            history_step=1,
            future_size=future_size,
        )
        if not output_all:  # here we choose to keep the last pred at each step
            prediction = cat_list_with_seq_idx(prediction, flatt_idx=-1)
    else:
        raise NotImplementedError("'%s' is not a valid run_method" % run_method)
    return prediction


def eval_predictors(
    predictors: List[Callable],
    trajectories: List[Trajectory],
    history_size: int,
    future_size: int,
    run_method: str = "step_every_timestamp",
    do_plotting: bool = True,
    saveplot_pattern: str = "%d_%s_prediction.png",
    saveplot_dir_path: str = str(Path("data") / "eval"),
) -> List[EvaluationSummary]:
    """Evaluate a list of predictors over a list of trajectories

    Args:
        predictors (List[Callable]): list of predictors
        trajectories (List[Trajectory]): list of trajectories
        history_size (int): size used as input for the predictor.
        future_size (int): size used as output for the predictor
        custom_step (int, optional): step used to loop over the input_tensor.
                If omitted, the step is determined by the run_method
                Defaults to None
        run_method (str, optional): method used to generate the predictions.
                "windowed" will predict with a step == history_size
                "step_every_timestamp" will predict with a step == 1
                Defaults to 'windowed'.
        do_plotting (bool, optional): if True will output the evaluation plots.
                Defaults to True.
        saveplot_pattern (str, optional): used to determine the path of the saved plot.
                Defaults to "%d_%s_prediction".
        saveplot_dir_path (str, optional): used to determine the path of the saved plot.
                Defaults to str(Path("data") / "eval").

    Returns:
        List[EvaluationSummary]: list of an evaluation summary for each predictor
    """
    # TODO: reject impossible values for history_size and future_size
    evaluation_results = [EvaluationSummary() for _ in predictors]
    logger.getChild(EVAL).info(
        f"Running evaluation for {len(predictors)} predictors"
        f" on {len(trajectories)} trajectories",
    )
    for t, trajectory in tqdm(enumerate(trajectories), desc="Trajectory n°"):
        predictions = []
        for p, predictor in tqdm(enumerate(predictors), desc="Predictor n°"):
            # prediction is made with chosen method and timed
            start = timeit.default_timer()
            prediction = run_predictor(
                predictor, trajectory.tensor, history_size, future_size, run_method
            )
            elapsed = (timeit.default_timer() - start) * 1000.0
            # we generate new evaluation results with the task metrics
            truth = trajectory.tensor[history_size:]
            # plot_truth_and_pred(trajectory.tensor,
            #                     truth,
            #                     prediction,
            #                     history_size,
            #                     "test.png")
            evaluation_results[p].results.append(
                EvaluationResult(trajectory.tensor, truth, prediction, elapsed)
            )
            # we plot a file per (predictor, trajectory) pair
            if do_plotting:
                savefig_path = str(
                    Path(saveplot_dir_path) / (saveplot_pattern % (t, predictor))
                )
                plot_trajectory_prediction(
                    trajectory, prediction, step=history_size, savefig_path=savefig_path
                )
            predictions.append(prediction)
        # we also plot a file per (predictor_list, trajectory) pair for a direct comparision
        # if do_plotting:
        #     savefig_path = str(Path(saveplot_dir_path) / (saveplot_pattern % (t, "all")))
        #     plot_multiple_predictors(trajectory, predictors, predictions,
        #                              step=history_size, savefig_path=savefig_path)
    for p, predictor in enumerate(predictors):
        predictor.log_evaluation_summary(evaluation_results[p])
    return evaluation_results


#  TODO: Think of some evaluator with a scaling future size, maybe like this but a bit redundant:
def evaluate_n_futures(
    predictors: List[Callable],
    trajectories: List[Trajectory],
    history_size: int,
    future_sizes: List[int],
    run_method: str = "step_every_timestamp",
    do_plotting: bool = True,
    saveplot_pattern: str = "%d_%s_prediction.png",
    saveplot_dir_path: str = str(Path("data") / "eval"),
):
    future_results = []
    for future_size in future_sizes:
        future_results.append(
            eval_predictors(
                predictors,
                trajectories,
                history_size,
                future_size,
                run_method,
                do_plotting,
                f"f{future_size}_" + saveplot_pattern,
                saveplot_dir_path,
            )
        )
    if do_plotting:
        for p, predictor in enumerate(predictors):
            for t, trajectory in enumerate(trajectories):
                per_future_pred = [
                    eval_result[p][t].pred for eval_result in future_results
                ]
                savefig_path = str(
                    Path(saveplot_dir_path)
                    / ("n_futures" + saveplot_pattern % (t, predictor))
                )
                plot_multiple_future(
                    future_sizes,
                    trajectory,
                    predictor,
                    per_future_pred,
                    step=history_size,
                    savefig_path=savefig_path,
                )
    return future_results
