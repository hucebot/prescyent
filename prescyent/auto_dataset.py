import json
from pathlib import Path
from typing import Union

from prescyent.dataset.config import MotionDatasetConfig
from prescyent.utils.logger import logger, DATASET
from prescyent.dataset import DATASET_MAP


class AutoDataset():

    @classmethod
    def build_from_config(cls, config: Union[str, Path, dict, MotionDatasetConfig]):
        if isinstance(config, (str, Path)):
            config = cls._get_config_from_path(Path(config))
        if isinstance(config, MotionDatasetConfig):
            config = config.dict()
        dataset_class_name = config.get("name", None)
        dataset_class = DATASET_MAP.get(dataset_class_name, None)
        if dataset_class is None:
            logger.error("Could not find a Dataset class matching %s",
                         dataset_class_name, group=DATASET)
            raise AttributeError(dataset_class_name)
        logger.info("Building new %s", dataset_class,
                    group=DATASET)
        return dataset_class(config=config)

    @classmethod
    def _get_config_from_path(cls, config_path: Path):
        if config_path.is_dir():
            config_path = config_path / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"No file or directory at {config_path}")
        try:
            with config_path.open(encoding="utf-8") as conf_file:
                return json.load(conf_file)
        except json.JSONDecodeError:
            logger.error("The provided config_file could not be loaded as Json", group=DATASET)
