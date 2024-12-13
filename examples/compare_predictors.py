"""use this script to compare multiple predictors over the same trajectories,
   plot and save the results"""

from argparse import ArgumentParser
from pathlib import Path
from typing import List, Union

import torch

from prescyent.auto_dataset import AutoDataset
from prescyent.auto_predictor import AutoPredictor
from prescyent.evaluator.plotting import plot_mpjpes
from prescyent.evaluator.runners import eval_predictors, dump_eval_summary_list
from prescyent.predictor import (
    ConstantDerivativePredictor,
    ConstantPredictor,
    DelayedPredictor,
    PredictorConfig,
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
    eval_folder = predictors_folder / "evaluation"
    eval_summary_list = eval_predictors(
        predictors=predictor_list,
        trajectories=dataset.trajectories.test,
        dataset_config=dataset.config,
        future_size=None,  # will default to the future size in dataset_config
        run_method="step_every_timestamp",  # we run the predictor at each frame and retain the last predicted frame to create the predicted trajectory
        do_plotting=True,  # you may want to disable theses or update the plotting function used as it is mainly made for low dim features (like Coordinates) and not too many points to plot
        saveplot_dir_path=eval_folder / "plots",
    )

    # use our function to save the multiple EvalSummaries
    dump_eval_summary_list(
        eval_summary_list, dump_dir=eval_folder, dump_prediction=True
    )
    print(f"Find evaluation summary and plots in:\n\t{eval_folder}")
