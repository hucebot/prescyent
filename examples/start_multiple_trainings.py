"""Use this script to train variations of your models"""

import argparse
import glob
import itertools
import json
import os
from pathlib import Path

from prescyent.utils.enums import (
    LearningTypes,
    LossFunctions,
    Scalers,
    TrajectoryDimensions,
)

from examples.train_from_config import train_from_config


DEFAULT_CONFIG_DIR = Path("data") / "configs"
FREQUENCY = 24
HISTORY_SIZE = 24
FUTURE_SIZE = 12
VARIATIONS = {
    "training_config.number_of_repetition": range(1),
    # SCALER
    "scaler_config.do_feature_wise_scaling": [True],
    "scaler_config.scale_rotations": [False],
    "scaler_config.scaler": [Scalers.STANDARDIZATION],
    "scaler_config.scaling_axis": [
        TrajectoryDimensions.FEATURE,
    ],
    # MODEL
    "model_config.name": [
        "MlpPredictor",
        "Seq2SeqPredictor",
        "siMLPe",
        "SARLSTMPredictor",  # Warning ! Can't be used in all conditions
    ],
    "model_config.loss_fn": [
        LossFunctions.MTRDLOSS,
    ],
    "model_config.deriv_on_last_frame": [
        False
    ],  # Warning ! Canno't be used in all conditions
    # ...
    # TRAINING
    "training_config.max_epochs": [200],
    # "training_config.devices": [1],
    # "training_config.accelerator": ["gpu"],
    "training_config.early_stopping_patience": [20],
    "training_config.use_auto_lr": [True],
    # DATASET
    "dataset_config.frequency": [FREQUENCY],
    "dataset_config.history_size": [HISTORY_SIZE],
    "dataset_config.future_size": [FUTURE_SIZE],
    "dataset_config.name": ["TeleopIcub"],
    "dataset_config.subsets": [["BottleTable"]],
    "dataset_config.batch_size": [256],
}

AUTO_REGRESSIVE_MODELS = ["SARLSTMPredictor"]
DATASET_IS_STATIC = True
MAX_WORKERS = 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="PrescyentBatchTraining",
        description="Start Batch training of Motion Predictors for prescyent",
    )
    parser.add_argument(
        "--config_dir",
        default=None,
        help="If provided, config will "
        "not be generated but loaded from the given directory",
    )
    args = parser.parse_args()

    # retreive directory with config variations from parsed args
    if args.config_dir:
        config_paths = sorted(
            [Path(p) for p in glob.glob(os.path.join(args.config_dir, "*.json"))]
        )
        print(f"Found {len(config_paths)} different training configs")
        if len(config_paths) < 1:
            print("Please provide a folder with compatible config files")
            exit(1)
        config_dict = json.load(open(config_paths[0], encoding="utf-8"))
    # generate config variations and write them on disk at default dir
    else:
        config_paths = []
        combinations = list(itertools.product(*list(VARIATIONS.values())))
        config_datas = [
            {list(VARIATIONS.keys())[i]: value for i, value in enumerate(combination)}
            for combination in combinations
        ]
        print(
            f"######################\nGenerated {len(combinations)} different training configs"
        )

        config_base_dir = DEFAULT_CONFIG_DIR
        if not config_base_dir.exists():
            config_base_dir.mkdir(parents=True)
        print(f"Writing config files in {config_base_dir}")
        for config_number, config_data in enumerate(config_datas):
            config_dict = {
                "model_config": dict(),
                "dataset_config": dict(),
                "training_config": dict(),
                "scaler_config": dict(),
            }
            del config_data["training_config.number_of_repetition"]
            for key, value in config_data.items():
                key1, key2 = key.split(".")
                config_dict[key1][key2] = value
            config_paths.append(config_base_dir / f"exp_config_{config_number}.json")
            config_dict["model_config"]["version"] = config_number
            if config_dict["scaler_config"]:
                config_dict["model_config"]["scaler_config"] = config_dict[
                    "scaler_config"
                ]
            del config_dict["scaler_config"]
            if config_dict["model_config"]["name"] in AUTO_REGRESSIVE_MODELS:
                config_dict["dataset_config"]["learning_type"] = LearningTypes.AUTOREG
            with open(config_paths[-1], "w", encoding="utf-8") as config_file:
                json.dump(config_dict, config_file, indent=4)

    # Start a new training per config file
    exp_path = (
        Path(__file__).parent.resolve()
        / "data"
        / "models"
        / "TeleopIcub"
        / f"{FREQUENCY}Hz_{HISTORY_SIZE}in_{FUTURE_SIZE}out"
    )
    for i, config_path in enumerate(config_paths):
        print(f"Training {i} starting...")
        train_from_config(config_path, rm_config=True, exp_path=exp_path)
        print(f"Training {i} ended.")

    # want lightning trainer to use multiple devices for one training
    # and it added unnecessary confusing behavior
    # For parallel training, you can call this script multiple times with different config_dir
