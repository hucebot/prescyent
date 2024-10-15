"""Standard class for motion datasets"""
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, List, Union, Type


import h5py
import json
import requests
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from prescyent.dataset.features import Features
from prescyent.dataset.hdf5_datasample import HDF5MotionDataSamples
from prescyent.dataset.hdf5_utils import get_dataset_keys
from prescyent.dataset.features.feature_manipulation import features_are_convertible_to
from prescyent.dataset.config import MotionDatasetConfig
from prescyent.dataset.datasamples import MotionDataSamples
from prescyent.dataset.trajectories.trajectories import Trajectories
from prescyent.dataset.trajectories.trajectory import Trajectory
from prescyent.utils.logger import logger, DATASET


def collate_context_fn(list_of_tensors: List[torch.Tensor]):
    """custom collate function to allow context_batch to be None, or a dict of batched tensors"""
    sample_batch = torch.stack([t[0] for t in list_of_tensors])
    truth_batch = torch.stack([t[2] for t in list_of_tensors])
    context_batch = {}
    context_batch = {
        c_name: torch.stack([context[1][c_name] for context in list_of_tensors])
        for c_name in list_of_tensors[0][1].keys()
    }
    return sample_batch, context_batch, truth_batch


def collate_batched_fn(list_of_tensors: List[torch.Tensor]):
    """simply return first occurence of tensors that are already batched by batch sampler"""
    return list_of_tensors[0]


class MotionDataset(LightningDataModule):
    """Base classe for all motion datasets"""

    config: MotionDatasetConfig
    """The config of the dataset"""
    config_class: Type[MotionDatasetConfig]
    """Class of the dataset config instance"""
    name: str
    """Name of the dataset, inherited from child class"""
    trajectories: Trajectories
    """Trajectories instance storing the trajectories per subset train, test and val"""
    train_datasample: Union[MotionDataSamples, HDF5MotionDataSamples]
    """Generated pairs for train"""
    test_datasample: Union[MotionDataSamples, HDF5MotionDataSamples]
    """Generated pairs for test"""
    val_datasample: Union[MotionDataSamples, HDF5MotionDataSamples]
    """Generated pairs for val"""

    def __init__(self, name: str) -> None:
        """
        Args:
            name (str): Name of the dataset, inherited from child class.
        """
        super().__init__()
        self.name = name
        if self.config.name is None:
            self.config.name = self.name

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
        """Method to generate the dataset.x_datasample used in the x_dataloader()"""
        self.update_trajectories_frequency(self.config.frequency)
        if self.config.convert_trajectories_beforehand:
            self.convert_trajectories_from_config()
        self.generate_samples(stage)

    def get_trajnames_from_hdf5(
        self,
        hdf5_data: h5py.File,
        tmp_hdf5_data: h5py.File,
    ) -> List[str]:
        # Create new temp hdf5 with features from config
        # Copy root attributes
        for attr in hdf5_data.attrs.keys():
            tmp_hdf5_data.attrs[attr] = hdf5_data.attrs[attr]
        all_keys = get_dataset_keys(hdf5_data)
        # Features
        all_feature_names = [key for key in all_keys if key[:16] == "tensor_features/"]
        for feat_name in all_feature_names:
            old_feat = hdf5_data[feat_name]
            feat = tmp_hdf5_data.create_dataset(feat_name, data=old_feat)
            for attr_name in old_feat.attrs.keys():
                feat.attrs[attr_name] = old_feat.attrs[attr_name]
        all_trajectory_names = [key for key in all_keys if key[-5:] == "/traj"]
        return all_trajectory_names

    def generate_samples(self, stage: str = None):
        logger.getChild(DATASET).debug(
            "Tensor pairs will be generated for a %s learning type",
            self.config.learning_type,
        )
        if stage is None or stage == "fit":
            if self.config.save_samples_on_disk and self.trajectories.train:
                self.train_datasample = HDF5MotionDataSamples(
                    self.trajectories.train, self.config
                )
            else:
                self.train_datasample = MotionDataSamples(
                    self.trajectories.train, self.config
                )
            logger.getChild(DATASET).info(
                "Train dataset has a size of %d", len(self.train_datasample)
            )
            if self.config.save_samples_on_disk and self.trajectories.val:
                self.val_datasample = HDF5MotionDataSamples(
                    self.trajectories.val, self.config
                )
            else:
                self.val_datasample = MotionDataSamples(
                    self.trajectories.val, self.config
                )
            logger.getChild(DATASET).info(
                "Val dataset has a size of %d", len(self.val_datasample)
            )
            if self.train_datasample:
                logger.getChild(DATASET).info(
                    "Generated (x,y) pairs with shapes (%s, %s)",
                    self.train_datasample[0][0].shape,
                    self.train_datasample[0][2].shape,
                )
        if stage is None or stage == "test" or stage == "predict":
            # We will predict on the test dataset also
            if self.config.save_samples_on_disk and self.trajectories.test:
                self.test_datasample = HDF5MotionDataSamples(
                    self.trajectories.test, self.config
                )
            else:
                self.test_datasample = MotionDataSamples(
                    self.trajectories.test, self.config
                )
            logger.getChild(DATASET).info(
                "Test dataset has a size of %d", len(self.test_datasample)
            )
            if self.test_datasample:
                logger.getChild(DATASET).info(
                    "Generated (x,y) pairs with shapes (%s, %s)",
                    self.test_datasample[0][0].shape,
                    self.test_datasample[0][2].shape,
                )

    def convert_trajectories_from_config(self):
        """Convert trajectories features to match required in and out features
        Here it is performed at setup instead of at runtime in the dataloader
        """
        if (
            self.config.in_features == self.config.out_features
            and self.config.in_features != self.trajectories.tensor_features
        ) or (
            self.config.in_features != self.config.out_features
            and self.config.in_features != self.trajectories.tensor_features
            and self.config.out_features != self.trajectories.tensor_features
        ):
            if features_are_convertible_to(
                self.config.in_features, self.config.out_features
            ):
                logger.getChild(DATASET).info(
                    "Converting trajectories features from %s to %s",
                    self.trajectories.tensor_features,
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
                    self.trajectories.tensor_features,
                    self.config.out_features,
                )
                for traj in self.trajectories.train:
                    traj.convert_tensor_features(self.config.out_features)
                for traj in self.trajectories.test:
                    traj.convert_tensor_features(self.config.out_features)
                for traj in self.trajectories.val:
                    traj.convert_tensor_features(self.config.out_features)

    def update_trajectories_frequency(self, target_frequency):
        """Update all trajectories frequency"""
        if self.trajectories.frequency != target_frequency:
            for traj in self.trajectories.train:
                traj.update_frequency(target_frequency)
            for traj in self.trajectories.test:
                traj.update_frequency(target_frequency)
            for traj in self.trajectories.val:
                traj.update_frequency(target_frequency)

    def train_dataloader(self) -> DataLoader:
        """the train torch dataloader with the MotionDataSamples"""
        try:
            if self.config.save_samples_on_disk:
                sampler = torch.utils.data.sampler.BatchSampler(
                    torch.utils.data.sampler.RandomSampler(self.train_datasample),
                    batch_size=self.config.batch_size,
                    drop_last=False,
                )
                return DataLoader(
                    self.train_datasample,
                    sampler=sampler,
                    num_workers=self.config.num_workers,
                    pin_memory=self.config.pin_memory,
                    persistent_workers=self.config.persistent_workers,
                    collate_fn=collate_batched_fn,
                )
            else:
                return DataLoader(
                    self.train_datasample,
                    batch_size=self.config.batch_size,
                    num_workers=self.config.num_workers,
                    pin_memory=self.config.pin_memory,
                    shuffle=True,
                    drop_last=False,
                    persistent_workers=self.config.persistent_workers,
                    collate_fn=collate_context_fn,
                )
        except AttributeError:
            logger.getChild(DATASET).error(
                "Pairs were not created for this datamodule. "
                + "Please make sure that you are using this dm through Lightning, "
                + "or call .prepare_data() and .setup() manually."
            )

    def test_dataloader(self) -> DataLoader:
        """the test torch dataloader with the MotionDataSamples"""
        try:
            if self.config.save_samples_on_disk:
                sampler = torch.utils.data.sampler.BatchSampler(
                    torch.utils.data.SequentialSampler(self.test_datasample),
                    batch_size=self.config.batch_size,
                    drop_last=False,
                )
                return DataLoader(
                    self.test_datasample,
                    sampler=sampler,
                    num_workers=self.config.num_workers,
                    pin_memory=self.config.pin_memory,
                    persistent_workers=self.config.persistent_workers,
                    collate_fn=collate_batched_fn,
                )
            else:
                return DataLoader(
                    self.test_datasample,
                    batch_size=self.config.batch_size,
                    num_workers=self.config.num_workers,
                    pin_memory=self.config.pin_memory,
                    shuffle=False,
                    drop_last=False,
                    persistent_workers=self.config.persistent_workers,
                    collate_fn=collate_context_fn,
                )
        except AttributeError:
            logger.getChild(DATASET).error(
                "Pairs were not created for this datamodule. "
                + "Please make sure that you are using this dm through Lightning, "
                + "or call .prepare_data() and .setup() manually."
            )
            raise

    def val_dataloader(self) -> DataLoader:
        """the val torch dataloader with the MotionDataSamples"""
        try:
            if self.config.save_samples_on_disk:
                sampler = torch.utils.data.sampler.BatchSampler(
                    torch.utils.data.SequentialSampler(self.val_datasample),
                    batch_size=self.config.batch_size,
                    drop_last=False,
                )
                return DataLoader(
                    self.val_datasample,
                    sampler=sampler,
                    num_workers=self.config.num_workers,
                    pin_memory=self.config.pin_memory,
                    persistent_workers=self.config.persistent_workers,
                    collate_fn=collate_batched_fn,
                )
            else:
                return DataLoader(
                    self.val_datasample,
                    batch_size=self.config.batch_size,
                    num_workers=self.config.num_workers,
                    pin_memory=self.config.pin_memory,
                    shuffle=False,
                    drop_last=False,
                    persistent_workers=self.config.persistent_workers,
                    collate_fn=collate_context_fn,
                )
        except AttributeError:
            logger.getChild(DATASET).error(
                "Pairs were not created for this datamodule. "
                + "Please make sure that you are using this dm through Lightning, "
                + "or call .prepare_data() and .setup() manually."
            )

    # We will predict on the test dataset also
    def predict_dataloader(self) -> DataLoader:
        """the predict torch dataloader with the MotionDataSamples"""
        try:
            if self.config.save_samples_on_disk:
                sampler = torch.utils.data.sampler.BatchSampler(
                    torch.utils.data.SequentialSampler(self.test_datasample),
                    batch_size=self.config.batch_size,
                    drop_last=False,
                )
                return DataLoader(
                    self.test_datasample,
                    sampler=sampler,
                    num_workers=self.config.num_workers,
                    pin_memory=self.config.pin_memory,
                    persistent_workers=self.config.persistent_workers,
                    collate_fn=collate_batched_fn,
                )
            else:
                return DataLoader(
                    self.test_datasample,
                    batch_size=self.config.batch_size,
                    num_workers=self.config.num_workers,
                    pin_memory=self.config.pin_memory,
                    shuffle=False,
                    drop_last=False,
                    persistent_workers=self.config.persistent_workers,
                    collate_fn=collate_context_fn,
                )
        except AttributeError:
            logger.getChild(DATASET).error(
                "Pairs were not created for this datamodule. "
                + "Please make sure that you are using this dm through Lightning, "
                + "or call .prepare_data() and .setup() manually."
            )

    @property
    def trajectories(self):
        if not hasattr(self, "_trajectories"):
            self.prepare_data()
        return self._trajectories

    @trajectories.setter
    def trajectories(self, trajectories):
        self._trajectories = trajectories

    @property
    def train_datasample(self):
        if not hasattr(self, "_train_datasample"):
            self.setup("fit")
        return self._train_datasample

    @train_datasample.setter
    def train_datasample(self, train_datasample):
        self._train_datasample = train_datasample

    @property
    def test_datasample(self):
        if not hasattr(self, "_test_datasample"):
            self.setup("test")
        return self._test_datasample

    @test_datasample.setter
    def test_datasample(self, test_datasample):
        self._test_datasample = test_datasample

    @property
    def val_datasample(self):
        if not hasattr(self, "_val_datasample"):
            self.setup("fit")
        return self._val_datasample

    @val_datasample.setter
    def val_datasample(self, val_datasample):
        self._val_datasample = val_datasample

    @property
    def frequency(self) -> int:
        return self.trajectories.frequency

    @property
    def num_points(self) -> int:
        return self.trajectories.num_points

    @property
    def num_dims(self) -> int:
        return self.trajectories.num_dims

    @property
    def tensor_features(self) -> Features:
        return self.trajectories.tensor_features

    @property
    def context_sizes(self) -> Dict[str, int]:
        return self.trajectories.context_sizes

    @property
    def context_size_sum(self) -> int:
        return self.trajectories.context_size_sum

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
