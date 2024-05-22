"""Use this script to train variations of your models"""

import argparse
import glob
import itertools
import json
import os
from pathlib import Path
from prescyent.evaluator.plotting import plot_mpjpe
from prescyent.predictor.constant_predictor import ConstantPredictor

from prescyent.train_from_config import train_from_config
from prescyent.dataset import AndyDataset, AndyDatasetConfig
from prescyent.dataset.features import CoordinateXYZ, RotationRep6D
from prescyent.utils.enums import Normalizations
from prescyent.utils.enums.loss_functions import LossFunctions


VARIATIONS = {
    "training_config.number_of_repetition": range(5),
    # MODEL
    "model_config.name": [
        "siMLPe",
    ],
    "model_config.hidden_size": [64, 128],
    "model_config.num_layers": [24, 48],
    "model_config.loss_fn": [LossFunctions.MSELOSS],
    "model_config.used_norm": [
        Normalizations.SPATIAL,
    ],
    "model_config.norm_on_last_input": [True, False],
    # ...
    # TRAINING
    "training_config.epoch": [200],
    "training_config.devices": [1],
    "training_config.accelerator": ["gpu"],
    "training_config.early_stopping_patience": [20],
    "training_config.use_auto_lr": [True],
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
                "training_config": dict(),
            }
            for key, value in config_data.items():
                key1, key2 = key.split(".")
                config_dict[key1][key2] = value
            config_paths.append(config_base_dir / f"exp_config_{config_number}.json")
            config_dict["model_config"]["version"] = config_number
            with open(config_paths[-1], "w", encoding="utf-8") as config_file:
                json.dump(config_dict, config_file, indent=4)
    features = [CoordinateXYZ(range(3)), RotationRep6D(range(3, 9))]
    dataset_config = AndyDatasetConfig(
        batch_size=258 * 8,
        subsampling_step=24,  # subsampling -> 240 Hz to 10Hz
        future_size=10,
        history_size=25,
        in_features=features,
        out_features=features,
        # in_points=[10],  # All points as input
        out_points=[10],  # only right hand as output
        make_joints_position_relative_to=0,  # Whole points coordinates are relative to PELVIS's position
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    dataset = AndyDataset(dataset_config)
    exp_path = (
        f"data/models/{dataset.DATASET_NAME}_ee"
        f"/h{dataset_config.history_size}_f{dataset_config.future_size}_{dataset.frequency}hz"
        f"/i{''.join([feat.__class__.__name__ for feat in  dataset_config.in_features])}_o{''.join([feat.__class__.__name__ for feat in  dataset_config.in_features])}"
    )

    # Test Baseline first
    constant = ConstantPredictor(dataset.config, str(exp_path))
    constant.test(dataset)
    plot_mpjpe(constant, dataset, savefig_dir_path=f"{constant.log_path}/")

    # Start a new training per config file
    for i, config_path in enumerate(config_paths):
        print(f"Training {i} starting...")
        train_from_config(
            config_path, rm_config=True, exp_path=exp_path, dataset=dataset
        )
        print(f"Training {i} ended.")
