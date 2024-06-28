"""Standard class for motion datasets"""
import zipfile
from pathlib import Path
from typing import Dict, List, Union, Type

import json
import requests
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from prescyent.dataset.features import Feature
from prescyent.dataset.features.feature_manipulation import features_are_convertible_to
from prescyent.dataset.config import MotionDatasetConfig
from prescyent.dataset.datasamples import MotionDataSamples
from prescyent.dataset.trajectories.trajectories import Trajectories
from prescyent.dataset.trajectories.trajectory import Trajectory
from prescyent.utils.logger import logger, DATASET


class MotionDataset(LightningDataModule):
    """Base classe for all motion datasets"""

    config: MotionDatasetConfig
    config_class: Type[MotionDatasetConfig]
    name: str
    trajectories: Trajectories
    train_datasample: MotionDataSamples
    test_datasample: MotionDataSamples
    val_datasample: MotionDataSamples

    def __init__(self, name: str, load_data_at_init: bool = True) -> None:
        super().__init__()
        self.name = name
        if self.config.name is None:
            self.config.name = self.name
        if load_data_at_init:
            self.prepare_data()
            self.setup()

    def __getitem__(self, index) -> Trajectory:
        return self.trajectories[index]

    def __len__(self) -> int:
        return len(self.trajectories)

    def __str__(self) -> str:
        return self.name

    def prepare_data(self):
        """Method to generates the dataset.trajectories"""
        raise NotImplementedError("This method must be implemented in the child class")

    def setup(self, stage: str = None):
        if self.config.convert_trajectories_beforehand:
            self.convert_trajectories_from_config()
        self.generate_samples(stage)

    def generate_samples(self, stage: str = None):
        logger.getChild(DATASET).debug(
            "Tensor pairs will be generated for a %s learning type",
            self.config.learning_type,
        )
        if stage is None or stage == "fit":
            self.train_datasample = MotionDataSamples(
                self.trajectories.train, self.config
            )
            logger.getChild(DATASET).info(
                "Train dataset has a size of %d", len(self.train_datasample)
            )
            self.val_datasample = MotionDataSamples(self.trajectories.val, self.config)
            logger.getChild(DATASET).info(
                "Val dataset has a size of %d", len(self.val_datasample)
            )
            logger.getChild(DATASET).info(
                "Generated (x,y) pairs with shapes (%s, %s)",
                self.train_datasample[0][0].shape,
                self.train_datasample[0][1].shape,
            )
        if stage is None or stage == "test" or stage == "predict":
            # We will predict on the test dataset also
            self.test_datasample = MotionDataSamples(
                self.trajectories.test, self.config
            )
            logger.getChild(DATASET).info(
                "Test dataset has a size of %d", len(self.test_datasample)
            )
            logger.getChild(DATASET).info(
                "Generated (x,y) pairs with shapes (%s, %s)",
                self.test_datasample[0][0].shape,
                self.test_datasample[0][1].shape,
            )

    def convert_trajectories_from_config(self):
        """Convert trajectories features to match required in and out features
        Here it is performed at setup instead of at runtime in the dataloader
        """
        if (
            self.config.in_features == self.config.out_features
            and self.config.in_features != self.trajectories.train[0].tensor_features
        ) or (
            self.config.in_features != self.config.out_features
            and self.config.in_features != self.trajectories.train[0].tensor_features
            and self.config.out_features != self.trajectories.train[0].tensor_features
        ):
            if features_are_convertible_to(
                self.config.in_features, self.config.out_features
            ):
                logger.getChild(DATASET).info(
                    "Converting trajectories features from %s to %s",
                    self.trajectories.train[0].tensor_features,
                    self.config.in_features,
                )
                for traj in self.trajectories.train:
                    traj.convert_tensor_features(self.config.in_features)
                for traj in self.trajectories.test:
                    traj.convert_tensor_features(self.config.in_features)
                for traj in self.trajectories.val:
                    traj.convert_tensor_features(self.config.in_features)
            elif features_are_convertible_to(
                self.config.out_features, self.config.in_features
            ):
                logger.getChild(DATASET).info(
                    "Converting trajectories features from %s to %s",
                    self.trajectories.train[0].tensor_features,
                    self.config.out_features,
                )
                for traj in self.trajectories.train:
                    traj.convert_tensor_features(self.config.out_features)
                for traj in self.trajectories.test:
                    traj.convert_tensor_features(self.config.out_features)
                for traj in self.trajectories.val:
                    traj.convert_tensor_features(self.config.out_features)

    def train_dataloader(self) -> DataLoader:
        try:
            return DataLoader(
                self.train_datasample,
                batch_size=self.config.batch_size,
                shuffle=self.config.shuffle_train,
                num_workers=self.config.num_workers,
                pin_memory=self.config.pin_memory,
                drop_last=self.config.drop_last,
                persistent_workers=self.config.persistent_workers,
            )
        except AttributeError:
            logger.getChild(DATASET).error(
                "Pairs were not created for this datamodule. "
                + "Please make sure that you are using this dm through Lightning, "
                + "or call .prepare_data() and .setup() manually."
            )

    def test_dataloader(self) -> DataLoader:
        try:
            return DataLoader(
                self.test_datasample,
                batch_size=self.config.batch_size,
                shuffle=self.config.shuffle_test,
                num_workers=self.config.num_workers,
                pin_memory=self.config.pin_memory,
                drop_last=False,
                persistent_workers=self.config.persistent_workers,
            )
        except AttributeError:
            logger.getChild(DATASET).error(
                "Pairs were not created for this datamodule. "
                + "Please make sure that you are using this dm through Lightning, "
                + "or call .prepare_data() and .setup() manually."
            )

    def val_dataloader(self) -> DataLoader:
        try:
            return DataLoader(
                self.val_datasample,
                batch_size=self.config.batch_size,
                shuffle=self.config.shuffle_val,
                num_workers=self.config.num_workers,
                pin_memory=self.config.pin_memory,
                drop_last=False,
                persistent_workers=self.config.persistent_workers,
            )
        except AttributeError:
            logger.getChild(DATASET).error(
                "Pairs were not created for this datamodule. "
                + "Please make sure that you are using this dm through Lightning, "
                + "or call .prepare_data() and .setup() manually."
            )

    # We will predict on the test dataset also
    def predict_dataloader(self) -> DataLoader:
        try:
            return DataLoader(
                self.test_datasample,
                batch_size=self.config.batch_size,
                shuffle=self.config.shuffle_test,
                num_workers=self.config.num_workers,
                pin_memory=self.config.pin_memory,
                drop_last=False,
                persistent_workers=self.config.persistent_workers,
            )
        except AttributeError:
            logger.getChild(DATASET).error(
                "Pairs were not created for this datamodule. "
                + "Please make sure that you are using this dm through Lightning, "
                + "or call .prepare_data() and .setup() manually."
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
    def tensor_features(self) -> List[Feature]:
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
            zip_path = Path(zip_path)
            zip_ref.extractall(zip_path.parent)
        logger.getChild(DATASET).info("Archive unziped at %s", zip_path.parent)
