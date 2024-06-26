"""module to run model and methods evaluation"""

from pathlib import Path
from typing import Callable, List, Union
import timeit

import torch
from tqdm import tqdm
from prescyent.dataset import Trajectory
from prescyent.dataset.config import MotionDatasetConfig
from prescyent.dataset.features import convert_tensor_features_to
from prescyent.evaluator.eval_result import EvaluationResult
from prescyent.evaluator.eval_summary import EvaluationSummary

from prescyent.evaluator.plotting import plot_trajectory_prediction
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
                (batch_len, seq_len, num_points, num_dims) or (seq_len, num_points, num_dims)
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
    dataset_config: MotionDatasetConfig,
    future_size=None,
    run_method: str = "step_every_timestamp",
    do_plotting: bool = True,
    saveplot_pattern: str = "%d_%s_prediction.png",
    saveplot_dir_path: str = str(Path("data") / "eval"),
) -> List[EvaluationSummary]:
    """Evaluate a list of predictors over a list of trajectories

    Args:
        predictors (List[Callable]): list of predictors
        trajectories (List[Trajectory]): list of trajectories
        history_size (int): size used as input for the predictor. (default is 1 second)
        future_size (int): size used as output for the predictor. (default is 1 second)
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
    if future_size is None:
        future_size = dataset_config.future_size
    evaluation_results = [EvaluationSummary() for _ in predictors]
    logger.getChild(EVAL).info(
        f"Running evaluation for {len(predictors)} predictors"
        f" on {len(trajectories)} trajectories",
    )
    for t, trajectory in tqdm(enumerate(trajectories), desc="Trajectory n°"):
        predictions = []
        for p, predictor in enumerate(predictors):
            # prediction is made with chosen method and timed
            start = timeit.default_timer()
            history = convert_tensor_features_to(
                trajectory.tensor,
                trajectory.tensor_features,
                dataset_config.in_features,
            )
            history = history[:, dataset_config.in_points]
            history = history[:, :, dataset_config.in_dims]
            prediction = run_predictor(
                predictor, history, dataset_config.history_size, future_size, run_method
            )
            elapsed = timeit.default_timer() - start
            truth = convert_tensor_features_to(
                trajectory.tensor,
                trajectory.tensor_features,
                dataset_config.out_features,
            )
            truth = truth[dataset_config.history_size :]
            truth = truth[:, dataset_config.out_points]
            truth = truth[:, :, dataset_config.out_dims]
            # we generate new evaluation results with the task metrics
            evaluation_results[p].results.append(
                EvaluationResult(
                    history,
                    truth,
                    prediction,
                    elapsed / trajectory.duration,
                    dataset_config.out_features,
                )
            )
            # we plot a file per (predictor, trajectory) pair
            if do_plotting:
                savefig_path = str(
                    Path(saveplot_dir_path) / (saveplot_pattern % (t, predictor))
                )
                trajectory.convert_tensor_features(dataset_config.out_features)
                plot_trajectory_prediction(
                    trajectory,
                    truth,
                    prediction,
                    step=dataset_config.history_size,
                    savefig_path=savefig_path,
                )
            predictions.append(prediction)
    for p, predictor in enumerate(predictors):
        predictor.log_evaluation_summary(evaluation_results[p])
    return evaluation_results
