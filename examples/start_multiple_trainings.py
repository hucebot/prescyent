"""Use this script to train variations of your models"""

import argparse
import copy
import glob
import itertools
import json
import os
from pathlib import Path

from train_from_config import train_from_config
from prescyent.auto_dataset import AutoDataset
from prescyent.utils.enums import LearningTypes


def get_dataset(size: int, config_dict):
    main_dataset = AutoDataset.build_from_config(config_dict["dataset_config"])
    for _ in range(size):
        yield copy.deepcopy(main_dataset)


def start_model_training(config_path, training_id, dataset=None):
    print(f"Starting Training {training_id}")
    train_from_config(config_path, rm_config=True, dataset=dataset)
    print(f"Training {training_id} ended.")


VARIATIONS = {
    "training_config.number_of_repetition": range(1),
    # MODEL
    "model_config.name": [
        "MlpPredictor",
        "Seq2SeqPredictor",
        "LinearPredictor",
        "siMLPe",
        "SARLSTMPredictor",
    ],
    "model_config.hidden_size": [32],
    "model_config.num_layers": [2],
    # "model_config.spatial_fc_only": [True, False],
    # "model_config.norm_axis": ["spatial", "temporal", "all"],
    # "model_config.dct": [True, False],
    # "model_config.dropout_value": [0, 0.1, 0.25],
    # "model_config.norm_on_last_input" : [False, True],
    # "model_config.do_lipschitz_continuation" : [False, True],
    # ...
    # TRAINING
    "training_config.epoch": [2],
    "training_config.devices": [1],
    # "training_config.accelerator": ["cpu"],
    "training_config.early_stopping_patience": [5],
    "training_config.use_auto_lr": [True],
    # DATASET
    "model_config.input_size": [10],
    "model_config.output_size": [10],
    "model_config.num_points": [3],
    "model_config.num_dims": [3],
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
    for i, config_path in enumerate(config_paths):
        start_model_training(config_path, i)

    # I removed any notion of multithreading for now as we often
    # want lightning trainer to use multiple devices for one training
    # and it added unnecessary confusing behavior
    # For parrallel training, you can call this script multiple times with different config_dir
