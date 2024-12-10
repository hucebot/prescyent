"""Class with methods to init any dataset from prescyent.dataset from its config file"""
import json
from pathlib import Path
from typing import Any, Dict, Union

from prescyent.dataset.dataset import TrajectoriesDataset, TrajectoriesDatasetConfig
from prescyent.utils.logger import logger, DATASET
from prescyent.dataset import DATASET_MAP


class AutoDataset:
    """Auto class building the requested Dataset class from a configuration and the dataset map {dataset_name_str: dataset_class}"""

    @classmethod
    def build_from_config(
        cls,
        config: Union[str, Path, dict, TrajectoriesDatasetConfig],
    ) -> TrajectoriesDataset:
        """Method to call upon to generate a new instance of a dataset from a config

        Args:
            config (Union[str, Path, dict, TrajectoriesDatasetConfig]): Path to a config json or actual config data

        Raises:
            AttributeError: if we cannot find the requested dataset class from the config file

        Returns:
            TrajectoriesDataset: an instance of the motion dataset from its configuration
        """

        if isinstance(config, (str, Path)):
            config = cls._get_config_from_path(Path(config))
        if isinstance(config, TrajectoriesDatasetConfig):
            config = config.model_dump()
        dataset_class_name = config.get("name", None)
        dataset_class = DATASET_MAP.get(dataset_class_name, None)
        if dataset_class is None:
            logger.getChild(DATASET).error(
                "Could not find a Dataset class matching %s",
                dataset_class_name,
            )
            raise AttributeError(dataset_class_name)
        logger.getChild(DATASET).info("Building new %s", dataset_class.__name__)
        return dataset_class(config=config)

    @classmethod
    def _get_config_from_path(cls, config_path: Path) -> Dict[str, Any]:
        """Method to resolve the config path, searching for .json files generated by the lib given the config_path attribute

        Args:
            config_path (Path): path of the file or directory

        Raises:
            FileNotFoundError: if no config file where found
            AttributeError: if found config file doesn't contain the expected data

        Returns:
            Dict[str, Any]: the config data
        """
        # try to load a dataset_config.json
        if config_path.is_dir():
            config_path = config_path / "dataset_config.json"
        # else try to load a config.json and retrieve dataset config of the predictor
        if not config_path.exists():
            config_path = config_path.parent / "config.json"
            if not config_path.exists():
                raise FileNotFoundError(f"No file or directory at {config_path}")
        try:
            if config_path.name == "config.json":
                with config_path.open(encoding="utf-8") as conf_file:
                    data = (
                        json.load(conf_file)
                        .get("model_config", {})
                        .get("dataset_config", None)
                    )
                    if data is None:
                        raise AttributeError(
                            f"No dataset_config found in file {config_path}"
                        )
                    return data
            else:
                with config_path.open(encoding="utf-8") as conf_file:
                    return json.load(conf_file)
        except json.JSONDecodeError:
            logger.getChild(DATASET).error(
                "The provided config_file could not be loaded as Json"
            )
