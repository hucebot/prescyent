import json
from pathlib import Path
from typing import Union

from prescyent.predictor.lightning.configs.module_config import ModuleConfig
from prescyent.utils.logger import logger, PREDICTOR
from prescyent.predictor import PREDICTOR_MAP


class AutoPredictor():

    @classmethod
    def load_from_config(cls, config: Union[str, Path, dict, ModuleConfig]):
        config_path = None
        if isinstance(config, (str, Path)):
            config_path = config
            config = cls._get_config_from_path(Path(config))
        else:
            if isinstance(config, ModuleConfig):
                config = config.dict()
            config_path = config.get("model_path", None)
        predictor_class_name = config.get("name", None)
        if predictor_class_name is None:
            predictor_class_name = config.get("model_config", {}).get("name")
        predictor_class = PREDICTOR_MAP.get(predictor_class_name, None)
        if predictor_class is None:
            logger.error("Could not find a predictor class matching %s",
                         predictor_class_name, group=PREDICTOR)
            raise AttributeError(predictor_class_name)
        if config_path is None:
            logger.error("Missing model path info")
        logger.info("Loading %s from %s", predictor_class.PREDICTOR_NAME, config_path,
                    group=PREDICTOR)
        return predictor_class(model_path=config_path)

    @classmethod
    def build_from_config(cls, config: Union[str, Path, dict, ModuleConfig]):
        if isinstance(config, (str, Path)):
            config = cls._get_config_from_path(Path(config))
        if isinstance(config, ModuleConfig):
            config = config.dict()
        predictor_class_name = config.get("name", None)
        if predictor_class_name is None:
            predictor_class_name = config.get("model_config", {}).get("name")
        predictor_class = PREDICTOR_MAP.get(predictor_class_name, None)
        if predictor_class is None:
            logger.error("Could not find a predictor class matching %s",
                         predictor_class_name, group=PREDICTOR)
            raise AttributeError(predictor_class_name)
        logger.info("Building new %s", predictor_class,
                    group=PREDICTOR)
        return predictor_class(config=config)

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
            logger.error("The provided config_file could not be loaded as Json", group=PREDICTOR)
