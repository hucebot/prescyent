"""Use this script to train variations of your models"""

import argparse
import copy
import glob
import itertools
import json
import os
from pathlib import Path

from prescyent.train_from_config import train_from_config
from prescyent.auto_dataset import AutoDataset
from prescyent.utils.enums import LearningTypes, Normalizations
from prescyent.utils.enums.loss_functions import LossFunctions


VARIATIONS = {
    "training_config.number_of_repetition": range(1),
    # MODEL
    "model_config.name": [
        "MlpPredictor",
        "Seq2SeqPredictor",
        "siMLPe",
        "SARLSTMPredictor",
    ],
    "model_config.hidden_size": [256],
    "model_config.num_layers": [4],
    "model_config.loss_fn": [
            LossFunctions.MPJPELOSS,
            ],
    "model_config.used_norm": [
        Normalizations.ALL,
        # Normalizations.SPATIAL,
        # Normalizations.TEMPORAL,
        # Normalizations.BATCH,
        # None
    ],
    # "model_config.spatial_fc_only": [True, False],
    # "model_config.dct": [True, False],
    # "model_config.dropout_value": [0, 0.1, 0.25],
    # "model_config.norm_on_last_input" : [True, False],
    # "model_config.do_lipschitz_continuation" : [False, True],
    # ...
    # TRAINING
    "training_config.epoch": [5],
    "training_config.devices": [1],
    # "training_config.accelerator": ["cpu"],
    "training_config.early_stopping_patience": [5],
    "training_config.use_auto_lr": [True],
    # DATASET
    "dataset_config.history_size": [10],
    "dataset_config.future_size": [10],
    "dataset_config.name": ["TeleopIcub"],
    "dataset_config.batch_size": [256],
    "dataset_config.num_workers": [4],
    "dataset_config.persistent_workers": [True],
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

    # retreive config variations with glob
    if args.config_dir:
        config_paths = sorted(
            [Path(p) for p in glob.glob(os.path.join(args.config_dir, "*.json"))]
        )
        print(f"Found {len(config_paths)} different training configs")
        if len(config_paths) < 1:
            print("Please provide a folder with compatible config files")
            exit(1)
        config_dict = json.load(open(config_paths[0], encoding="utf-8"))
    # generate config variations and write them
    else:
        config_paths = []
        combinations = list(itertools.product(*list(VARIATIONS.values())))
        config_datas = [
            {list(VARIATIONS.keys())[i]: value for i, value in enumerate(combination)}
            for combination in combinations
        ]
        print(f"Generated {len(combinations)} different training configs")

        config_base_dir = Path("data") / "configs"
        if not config_base_dir.exists():
            config_base_dir.mkdir(parents=True)
        print(f"Writing config files in {config_base_dir}")
        for config_number, config_data in enumerate(config_datas):
            config_dict = {
                "model_config": dict(),
                "dataset_config": dict(),
                "training_config": dict(),
            }
            for key, value in config_data.items():
                key1, key2 = key.split(".")
                config_dict[key1][key2] = value
            config_paths.append(config_base_dir / f"exp_config_{config_number}.json")
            config_dict["model_config"]["version"] = config_number
            if config_dict["model_config"]["name"] in AUTO_REGRESSIVE_MODELS:
                config_dict["dataset_config"]["learning_type"] = LearningTypes.AUTOREG
            with open(config_paths[-1], "w", encoding="utf-8") as config_file:
                json.dump(config_dict, config_file, indent=4)

    # Start a new training per config file
    exp_path = "data/models/TeleopIcub/10Hz_1s/"
    for i, config_path in enumerate(config_paths):
        print(f"Training {i} starting...")
        train_from_config(config_path, rm_config=True, exp_path=exp_path)
        print(f"Training {i} ended.")

    # I removed any notion of multithreading for now as we often
    # want lightning trainer to use multiple devices for one training
    # and it added unnecessary confusing behavior
    # For parrallel training, you can call this script multiple times with different config_dir
