import argparse
import concurrent.futures
import copy
import glob
import itertools
import json
import os
from pathlib import Path
from prescyent.auto_dataset import AutoDataset

from train_from_config import train_from_config


def get_dataset(size: int, config_dict):
    main_dataset = AutoDataset.build_from_config(config_dict["dataset_config"])
    for _ in range(size):
        yield copy.deepcopy(main_dataset)


def start_model_training(config_path, thread_id, dataset=None):
    print(f"Starting Thread {thread_id}")
    train_from_config(config_path, rm_config=True, dataset=dataset)
    print(f"Thread {thread_id} ended.")


VARIATIONS = {
    # "training_config.number_of_repeat": range(3),
    # MODEL
    "model_config.input_size": [50],
    "model_config.output_size": [10],
    "model_config.feature_size": [66],
    # "model_config.name": ["MlpPredictor", "Seq2SeqPredictor", "LinearPredictor"],
    "model_config.name": ["siMLPe"],
    "model_config.hidden_size": [64, 128, 256],
    "model_config.spatial_fc_only": [True, False],
    "model_config.norm_axis": ["spatial", "temporal", "all"],
    "model_config.dct": [True, False],
    "model_config.num_layers": [12, 48, 96],
    "model_config.dropout_value": [0, 0.1, 0.25],
    # "model_config.norm_on_last_input" : [False, True],
    # "model_config.do_lipschitz_continuation" : [False, True],
    # TRAINING
    "training_config.epoch": [100],
    "training_config.devices": [1],
    # "training_config.accelerator": ["cpu"],
    "training_config.early_stopping_patience": [5],
    "training_config.use_auto_lr": [True],
    # DATASET
    "dataset_config.name": ["H36M"],
    "dataset_config.batch_size": [256],
    "dataset_config.num_workers": [2],
    "dataset_config.persistent_workers": [True],
}


DATASET_IS_STATIC = True
MAX_WORKERS = 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="PrescyentBatchTraining",
        description="Start Batch training of Motion Predictors for prescyent",
    )
    parser.add_argument(
        "--config_folder",
        default=None,
        help="If provided, config will "
        "not be generated but loaded from the given directory",
    )
    args = parser.parse_args()

    # retreive config variations with glob
    if args.config_folder:
        config_paths = sorted(
            [Path(p) for p in glob.glob(os.path.join(args.config_folder, "*.json"))]
        )
        print(f"Found {len(config_paths)} different training configs")
        if len(config_paths) < 1:
            print("Please provide a folder with compatible config files")
            exit(1)
        config_dict = json.load(open(config_paths[0], encoding="utf-8"))
    # generate config variations and write them
    else:
        config_paths = []
        combinations = [p for p in itertools.product(*list(VARIATIONS.values()))]
        config_datas = [
            {list(VARIATIONS.keys())[i]: value for i, value in enumerate(combination)}
            for combination in combinations
        ]
        print(f"Generated {len(combinations)} different training configs")

        config_base_dir = Path("data") / "configs"
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
            with open(config_paths[-1], "w", encoding="utf-8") as config_file:
                json.dump(config_dict, config_file, indent=4)

args = [config_paths, range(len(config_paths))]
# If dataset is static, we win a few minutes by reusing it instead of re init in each threads
if DATASET_IS_STATIC:
    dataset_generator = get_dataset(len(config_paths), config_dict)
    args.append(dataset_generator)
# start_model_training(config_paths[0], range(len(config_paths))[0], dataset=next(dataset_generator))
with concurrent.futures.ThreadPoolExecutor(
    max_workers=MAX_WORKERS, thread_name_prefix="prescyent_training"
) as executor:
    executor.map(start_model_training, *args)
