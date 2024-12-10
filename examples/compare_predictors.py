"""use this script to compare multiple predictors over the same trajectories, plot and save the results"""

from argparse import ArgumentParser
from pathlib import Path
from typing import List, Union

import torch

from prescyent.auto_dataset import AutoDataset
from prescyent.auto_predictor import AutoPredictor
from prescyent.evaluator.plotting import plot_mpjpes
from prescyent.evaluator.runners import eval_predictors
from prescyent.evaluator.eval_summary import EvaluationSummary
from prescyent.predictor import (
    ConstantDerivativePredictor,
    ConstantPredictor,
    DelayedPredictor,
    PredictorConfig,
)


def dump_eval_summary_list(
    eval_summaries: List[EvaluationSummary],
    dump_dir: Union[Path, str] = "data/eval/",
    dump_prediction: bool = False,
):
    """method to create a csv file to summarize multiple eval summaries

    Args:
        eval_summaries (List[EvaluationSummary]): list of the evaluation summaries of each predictor
        dump_dir (Union[Path, str], optional): folder where the files are created. Defaults to "data/eval/".
        dump_prediction (bool, optional): if true, saves the predictions and truth under .pt format. Defaults to False.
    """
    if isinstance(dump_dir, str):
        dump_dir = Path(dump_dir)
    with (dump_dir / "eval_summary.csv").open("w", encoding="utf-8") as csv_file:
        headers = eval_summaries[0].headers
        csv_file.write(",".join(headers) + "\n")
        for eval_summary in eval_summaries:
            csv_file.write(",".join(eval_summary.as_array()) + "\n")
        print(f"Saved Evaluation Summary here: {csv_file}")
    if dump_prediction:
        for traj_id, _ in enumerate(eval_summaries[0].results):
            traj_dir = dump_dir / eval_summaries[0].results[traj_id].traj_name
            traj_dir.mkdir(parents=True, exist_ok=True)
            print(f"Saving prediction dumps here: {traj_dir}")
            torch.save(eval_summaries[0].results[traj_id].truth, traj_dir / "truth.pt")
            for pred_id, _ in enumerate(eval_summaries):
                torch.save(
                    eval_summaries[pred_id].results[traj_id].pred,
                    traj_dir
                    / f"{eval_summaries[pred_id].predictor_name}_future_{eval_summaries[pred_id].predicted_future}.pt",
                )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "dataset_config",
        help="Path to the config of the dataset (or to the predictor with the dataset_config you want to load)",
    )
    parser.add_argument(
        "predictors_folder",
        help="Path where all found predictors will be loaded and compared",
    )
    parser.add_argument(
        "--delayed_baseline",
        action="store_true",
        help="if true add the delayed_baseline to the tested predictors",
    )
    parser.add_argument(
        "--constant_baseline",
        action="store_true",
        help="if true add the constant_baseline to the tested predictors",
    )
    parser.add_argument(
        "--constant_derivative_baseline",
        action="store_true",
        help="if true add the constant_derivative_baseline to the tested predictors",
    )

    args = parser.parse_args()

    dataset_config = args.dataset_config
    predictors_folder = args.predictors_folder
    delayed_baseline = args.delayed_baseline
    constant_baseline = args.constant_baseline
    constant_derivative_baseline = args.constant_derivative_baseline

    # load dataset from config
    dataset = AutoDataset.build_from_config(dataset_config)

    # instantiate all found models from given directory
    predictors_folder = Path(predictors_folder)
    predictor_configs = predictors_folder.rglob("config.json")
    predictor_list = [
        AutoPredictor.load_pretrained(predictor_path)
        for predictor_path in predictor_configs
    ]

    # instanciate requested baselines
    if constant_derivative_baseline:
        predictor_list.append(
            ConstantDerivativePredictor(PredictorConfig(dataset_config=dataset.config))
        )
    if constant_baseline:
        predictor_list.append(
            ConstantPredictor(PredictorConfig(dataset_config=dataset.config))
        )
    if delayed_baseline:
        predictor_list.append(
            DelayedPredictor(PredictorConfig(dataset_config=dataset.config))
        )

    # plot comparative mpjpes on the same plot here
    plot_mpjpes(
        predictors=predictor_list, dataset=dataset, savefig_dir_path=predictors_folder
    )

    # use this huge method that runs each predictor over the passed trajectories
    # and save each result as EvaluationResult into EvaluationSummary for each predictor.
    # there are options to change future size and to plot each result
    eval_summaries = eval_predictors(
        predictors=predictor_list,
        trajectories=dataset.trajectories.test,
        dataset_config=dataset.config,
        future_size=None,  # will default to the future size in dataset_config
        run_method="step_every_timestamp",  # we run the predictor at each frame and retain the last predicted frame to create the predicted trajectory
        do_plotting=True,  # you may want to disable theses or update the plotting function used as it is mainly made for low dim features (like Coordinates) and not too many points to plot
        saveplot_dir_path=predictors_folder / "plots",
    )

    # use our function to save the multiple EvalSummaries
    dump_eval_summary_list(
        eval_summaries, dump_dir=predictors_folder / "plots", dump_prediction=True
    )
