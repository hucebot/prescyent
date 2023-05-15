import concurrent.futures
import itertools
import json
from pathlib import Path
from prescyent.auto_dataset import AutoDataset

from train_from_config import train_from_config


def start_model_training(config_path, thread_id, dataset=None):
    print(f"Starting Thread {thread_id}")
    train_from_config(config_path, rm_config=True, dataset=dataset)
    print(f"Thread {thread_id} ended.")

VARIATIONS = {
    "training_config.number_of_repeat": range(3),
# MODEL
    "model_config.input_size": [50],
    "model_config.output_size": [10],
    "model_config.feature_size": [66],
    # "model_config.name": ["MlpPredictor", "Seq2SeqPredictor", "LinearPredictor"],
    "model_config.name": ["LinearPredictor"],
    # "model_config.hidden_size": [64, 128],
    # "model_config.num_layers": [2, 3, 4],
    "model_config.dropout_value": [0, 0.1, 0.25],
    "model_config.norm_on_last_input" : [False, True],
    "model_config.do_lipschitz_continuation" : [False, True],
# TRAINING
    "training_config.epoch": [100],
    "training_config.devices": [1],
    "training_config.use_auto_lr": [True],
# DATASET
    "dataset_config.name": ["H36M"],
    "dataset_config.batch_size": [2048],
    "dataset_config.num_workers": [2],
    "dataset_config.pin_workers": [True]
}


DATASET_IS_STATIC = True
MAX_WORKERS = 16

combinations = [p for p in itertools.product(*list(VARIATIONS.values()))]
config_datas = [{list(VARIATIONS.keys())[i]: value for i, value in enumerate(combination)} for combination in combinations]
print(f"Generated {len(combinations)} different training configs")

config_paths = list()
config_base_dir = Path("data") / "configs"
print(f"Writing config files in {config_base_dir}")
for config_number, config_data in enumerate(config_datas):
    config_dict = {"model_config": dict(),
                   "dataset_config": dict(),
                   "training_config": dict()}
    for key, value in config_data.items():
        key1, key2 = key.split(".")
        config_dict[key1][key2] = value
    config_paths.append(config_base_dir / f"exp_config_{config_number}.json")
    with open(config_paths[-1], "w", encoding="utf-8") as config_file:
        json.dump(config_dict, config_file, indent=4)
args = [config_paths, range(len(config_paths))]
# If dataset is static, we win a few minutes by reusing it instead of re init in each threads
if DATASET_IS_STATIC:
    dataset = AutoDataset.build_from_config(config_dict["dataset_config"])
    args.append([dataset for i in range(len(config_paths))])
# start_model_training(config_paths[0], range(len(config_paths))[0], dataset=dataset)
with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS, thread_name_prefix="prescyent_training") as executor:
        executor.map(start_model_training, *args)
