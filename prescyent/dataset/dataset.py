"""Standard class for motion datasets"""
import zipfile
from pathlib import Path
from typing import Dict, List, Union, Type

import json
import requests
from torch.utils.data import Dataset, DataLoader

import prescyent.dataset.features as tensor_features
from prescyent.dataset.config import MotionDatasetConfig
from prescyent.dataset.datasamples import MotionDataSamples
from prescyent.dataset.trajectories.trajectories import Trajectories
from prescyent.dataset.trajectories.trajectory import Trajectory
from prescyent.utils.enums import LearningTypes
from prescyent.utils.logger import logger, DATASET


class MotionDataset(Dataset):
    """Base classe for all motion datasets"""

    config: MotionDatasetConfig
    config_class: Type[MotionDatasetConfig]
    name: str
    trajectories: Trajectories
    train_datasample: MotionDataSamples
    test_datasample: MotionDataSamples
    val_datasample: MotionDataSamples

    def __init__(self, name: str) -> None:
        logger.getChild(DATASET).debug(
            "Tensor pairs will be generated for a %s learning type",
            self.config.learning_type,
        )
        if self.config.in_points is None:
            self.config.in_points = list(range(self.trajectories.train[0].shape[1]))
        if self.config.out_points is None:
            self.config.out_points = list(range(self.trajectories.train[0].shape[1]))
        if self.config.in_features is None:
            self.config.in_features = self.trajectories.train[0].tensor_features
        if self.config.out_features is None:
            self.config.out_features = self.trajectories.train[0].tensor_features
        # change de feats of each trajectory if they don't match in_feat and out_feat
        if (
            self.config.in_features == self.config.out_features
            and self.config.in_features != self.trajectories.train[0].tensor_features
        ) or (
            self.config.in_features != self.config.out_features
            and self.config.in_features != self.trajectories.train[0].tensor_features
            and self.config.out_features != self.trajectories.train[0].tensor_features
        ):
            for traj in self.trajectories.train:
                traj.convert_tensor_features(self.config.in_features)
            for traj in self.trajectories.test:
                traj.convert_tensor_features(self.config.in_features)
            for traj in self.trajectories.val:
                traj.convert_tensor_features(self.config.in_features)
        self.name = name
        self.train_datasample = MotionDataSamples(self.trajectories.train, self.config)
        logger.getChild(DATASET).info(
            "Train dataset has a size of %d", len(self.train_datasample)
        )
        self.test_datasample = MotionDataSamples(self.trajectories.test, self.config)
        logger.getChild(DATASET).info(
            "Test dataset has a size of %d", len(self.test_datasample)
        )
        self.val_datasample = MotionDataSamples(self.trajectories.val, self.config)
        logger.getChild(DATASET).info(
            "Val dataset has a size of %d", len(self.val_datasample)
        )
        if self.config.learning_type == LearningTypes.SEQ2ONE:
            self.config.future_size = 1

    def __getitem__(self, index) -> Trajectory:
        return self.trajectories[index]

    def __len__(self) -> int:
        return len(self.trajectories)

    def __str__(self) -> str:
        return self.name

    @property
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_datasample,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            drop_last=self.config.drop_last,
            persistent_workers=self.config.persistent_workers,
        )

    @property
    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_datasample,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=1,
            pin_memory=self.config.pin_memory,
            drop_last=False,
            persistent_workers=False,
        )

    @property
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_datasample,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            drop_last=False,
            persistent_workers=self.config.persistent_workers,
        )

    @property
    def frequency(self) -> int:
        return self.trajectories.train[0].frequency

    @property
    def num_points(self) -> int:
        return self.trajectories.train[0].shape[1]

    @property
    def num_dims(self) -> int:
        return self.trajectories.train[0].shape[2]

    @property
    def tensor_features(self) -> List[tensor_features.Feature]:
        return self.trajectories.train[0].tensor_features

    @property
    def feature_size(self) -> int:
        return self.num_dims * self.num_points

    def _init_from_config(
        self,
        config: Union[Dict, None, str, Path, MotionDatasetConfig],
        config_class: Type[MotionDatasetConfig],
    ) -> None:
        self.config_class = config_class
        if isinstance(config, dict):  # use pydantic for dict verification
            config = config_class(**config)
        elif config is None:  # use default config if none
            logger.getChild(DATASET).info(
                "Using default config because none was provided"
            )
            config = config_class()
        elif isinstance(config, str):  # load from a string
            config = Path(config)
        config = self._load_config(config)
        assert isinstance(config, config_class)  # check our config type
        self.config = config

    def _load_config(self, config) -> MotionDatasetConfig:
        if isinstance(config, Path) or isinstance(config, str):  # load from a Path
            logger.getChild(DATASET).info("Loading config from %s", config)
            # serialize features manually
            with open(config, encoding="utf-8") as conf_file:
                data = json.load(conf_file)
            # if data.get("in_features", None):
            #     data["in_features"] = [getattr(tensor_features, feature["name"])(feature["ids"]) for feature in data["in_features"]]
            # if data.get("out_features", None):
            #     data["out_features"] = [getattr(tensor_features, feature["name"])(feature["ids"]) for feature in data["out_features"]]
            return self.config_class(**data)
        return config

    def save_config(self, save_path: Path) -> None:
        # check if parent folder exist, or create it
        if isinstance(save_path, str):
            save_path = Path(save_path)
        if not save_path.parent.exists():
            save_path.parent.mkdir(parents=True, exist_ok=True)
        logger.getChild(DATASET).info("Saving config to %s", save_path)
        config_dict = self.config.model_dump(exclude_defaults=True)
        config_dict["name"] = self.name
        with save_path.open("w", encoding="utf-8") as conf_file:
            logger.getChild(DATASET).debug(config_dict)
            json.dump(config_dict, conf_file, indent=4, sort_keys=True)

    def _download_files(self, url, path) -> None:
        """get the dataset files from an url"""
        logger.getChild(DATASET).info("Downloading data from %s", url)
        data = requests.get(url, timeout=10)
        path = Path(path)
        if path.is_dir():
            path = path / "downloaded_data.zip"
        path.parent.mkdir(parents=True, exist_ok=True)
        logger.getChild(DATASET).info("Saving data to %s", path)
        with path.open("wb") as pfile:
            pfile.write(data.content)

    def _unzip(self, zip_path: str) -> None:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(zip_path.replace(".zip", ""))
        logger.getChild(DATASET).info(
            "Archive unziped at %s", zip_path.replace(".zip", "")
        )
