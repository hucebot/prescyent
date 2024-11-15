"""script to train a model from args using auto_predictor"""

import json
from argparse import ArgumentParser
import os
from pathlib import Path
import socket

from prescyent.auto_predictor import AutoPredictor
from prescyent.auto_dataset import AutoDataset
from prescyent.dataset.dataset import TrajectoriesDataset
from prescyent.evaluator.plotting import plot_mpjpe
from prescyent.predictor.lightning.configs.training_config import TrainingConfig
from prescyent.evaluator.runners import eval_predictors
from prescyent.utils.logger import logger, TRAINING, TESTING


DEFAULT_EXP_PATH = str(Path("data") / "models" / "exp")


def train_from_config(
    config_path: Path,
    rm_config: bool = False,
    dataset: TrajectoriesDataset = None,
    exp_path: str = DEFAULT_EXP_PATH,
):
    """Start a training for a predictor instanciated through a config file

    Args:
        config_path (Path): path to the config file
        rm_config (bool, optional): if true, config file is removed when trained predictor is saved. Defaults to False.
        dataset (_type_, optional): if not None, this instance will be used instead of loading a new dataset from the config. Defaults to None.
        exp_path (_type_, optional): custom path to save all models. Defaults to DEFAULT_EXP_PATH.
    """

    if not config_path.exists():
        logger.getChild(TRAINING).error("The provided config_file does not exist.")
        exit(1)
    try:
        with config_path.open(encoding="utf-8") as config_file:
            config_dict = json.load(config_file)
    except json.JSONDecodeError as e:
        logger.getChild(TRAINING).error(
            "The provided config_file could not be loaded as Json, please check your file"
        )
        logger.getChild(TRAINING).error(e)
        exit(1)

    # Get subparts of the global config
    model_config = config_dict.get("model_config", {})
    training_config = config_dict.get("training_config", {})
    dataset_config = config_dict.get("dataset_config", {})
    if not dataset_config:
        dataset_config = model_config.get("dataset_config", {})
    logger.getChild(TRAINING).info(f"Using model config: {model_config}")
    logger.getChild(TRAINING).info(f"Using training config: {training_config}")
    logger.getChild(TRAINING).info(f"Using dataset config: {dataset_config}")

    # Validate config content and create Dataset and Predictor
    if not dataset:
        dataset = AutoDataset.build_from_config(dataset_config)
    model_config["dataset_config"] = dataset.config
    predictor = AutoPredictor.build_from_config(model_config)
    training_config = TrainingConfig(**training_config)

    model_dir = Path(
        f"{exp_path}/{predictor.name}/version_{predictor.version}_{socket.gethostname()}"
    )
    predictor.save(model_dir)
    dataset.save_config(model_dir / "dataset_config.json")

    # Launch training
    logger.getChild(TRAINING).info("Training starts...")
    predictor.train(dataset, training_config)

    logger.getChild(TRAINING).info(f"Model directory: {model_dir}")
    predictor.save()
    if rm_config:
        os.remove(str(config_path))

    logger.getChild(TESTING).info("Testing predictor...")
    predictor.test(dataset)
    predictor.free_trainer()
    plot_mpjpe(predictor, dataset, savefig_dir_path=predictor.log_path)
    eval_predictors(
        predictors=[predictor],
        trajectories=dataset.trajectories.test,
        dataset_config=dataset.config,
        do_plotting=False,
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "config_path",
        type=str,
        help="path to the config file used to build model and dataset. Default if linear_teleop.json as an example",
    )
    parser.add_argument(
        "--rm_config",
        action="store_true",
        default=False,
        help="If provided, the config file will be removed after training. Default is False",
    )

    args = parser.parse_args()
    train_from_config(config_path=Path(args.config_path), rm_config=args.rm_config)
