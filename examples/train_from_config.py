"""script to train a model from args using auto_predictor"""
import json
from argparse import ArgumentParser
import os
from pathlib import Path
import socket

from prescyent.auto_predictor import AutoPredictor
from prescyent.auto_dataset import AutoDataset
from prescyent.predictor.lightning.configs.training_config import TrainingConfig
from prescyent.experimental.simlpe.benchmark import run_benchmark


def train_from_config(config_path: Path, rm_config: bool = False, dataset=None):
    if not config_path.exists():
        print("The provided config_file does not exist.")
        exit(1)
    try:
        with config_path.open(encoding="utf-8") as config_file:
            config_dict = json.load(config_file)
    except json.JSONDecodeError as e:
        print("The provided config_file could not be loaded as Json, please check your file")
        print(e)
        exit(1)

    # Get subparts of the global config
    model_config = config_dict.get("model_config", {})
    training_config = config_dict.get("training_config", {})
    dataset_config = config_dict.get("dataset_config", {})
    print(f"Using model config: {model_config}")
    print(f"Using training config: {training_config}")
    print(f"Using dataset config: {dataset_config}")

    # Validate config content and create Dataset and Predictor
    if not dataset:
        dataset = AutoDataset.build_from_config(dataset_config)
    predictor = AutoPredictor.build_from_config(model_config)
    training_config = TrainingConfig(**training_config)

    # Launch training
    print("Training...")
    predictor.train(dataset.train_dataloader, training_config,
                    dataset.val_dataloader)

    # Save the predictor, and configs
    model_dir = Path(f"data/models/exp/{predictor.name}/version_{predictor.version}_{socket.gethostname()}")
    print("model directory:", model_dir)
    predictor.save(model_dir)
    dataset.save_config(model_dir / 'dataset_config.json')
    if rm_config:
        os.remove(str(config_path))

    # Test so that we know how good we are
    print("Testing...")
    predictor.test(dataset.test_dataloader)
    # also run and log SiMLPe benchmark when possible
    if dataset.name == "H36M":
        try:
            print("Running SiMLPe benchmark...")
            res = run_benchmark(predictor)
            predictor.log_metrics(res)
        except RuntimeError as e:
            print(e)
            print("Aborting SiMLPe benchmark")


if __name__ == "__main__":
    parser = ArgumentParser()
    default_config = Path("examples") / "configs" / "linear_teleop.json"
    parser.add_argument("--config_path", type=str, default=str(default_config),
                        help="path to the config file used to build model and dataset. Default if linear_teleop.json as an example")
    parser.add_argument("--rm_config", action="store_true", default=False,
                        help="If provided, the config file will be removed after training. Default is False")

    args = parser.parse_args()
    config_path = Path(args.config_path)
    train_from_config(config_path=config_path, rm_config=args.rm_config)
