import json
from pathlib import Path
from typing import Union

from prescyent.dataset.dataset import MotionDataset, MotionDatasetConfig
from prescyent.utils.logger import logger, DATASET
from prescyent.dataset import DATASET_MAP


class AutoDataset:
    @classmethod
    def build_from_config(
        cls,
        config: Union[str, Path, dict, MotionDatasetConfig],
        load_data_at_init: bool = True,
    ) -> MotionDataset:
        if isinstance(config, (str, Path)):
            config = cls._get_config_from_path(Path(config))
        if isinstance(config, MotionDatasetConfig):
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
        return dataset_class(config=config, load_data_at_init=load_data_at_init)

    @classmethod
    def _get_config_from_path(cls, config_path: Path):
        if config_path.is_dir():
            dir_path = config_path
            config_path = config_path / "dataset_config.json"
        if not config_path.exists():
            config_path = dir_path / "config.json"
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
