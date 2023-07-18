import json
from pathlib import Path
from typing import Tuple, Union

from prescyent.predictor.base_predictor import BasePredictor
from prescyent.predictor.constant_predictor import ConstantPredictor
from prescyent.predictor.lightning.configs.module_config import ModuleConfig
from prescyent.utils.errors import PredictorNotFound, PredictorUnprocessable
from prescyent.utils.logger import logger, PREDICTOR
from prescyent.predictor import PREDICTOR_MAP


def get_predictor_from_path(predictor_path: str = None) -> BasePredictor:
    if predictor_path:
        return AutoPredictor.load_from_config(predictor_path)
    else:
        return ConstantPredictor()


def get_predictor_infos(config):
    predictor_class_name = config.get("name", None)
    if predictor_class_name is None:
        predictor_class_name = config.get("model_config", {}).get("name")
    predictor_class = PREDICTOR_MAP.get(predictor_class_name, None)
    if predictor_class is None:
        logger.error(
            "Could not find a predictor class matching %s",
            predictor_class_name,
            group=PREDICTOR,
        )
        raise AttributeError(predictor_class_name)
    return predictor_class


class AutoPredictor:
    @classmethod
    def preprocess_config_attribute(cls, config) -> Tuple[dict, str]:
        if isinstance(config, (str, Path)):
            return cls._get_config_from_path(Path(config)), str(config)
        elif isinstance(config, ModuleConfig):
            return config.dict(), None
        elif isinstance(config, dict):
            return config, None
        else:
            raise NotImplementedError('Check your attr "config"\'s type')

    @classmethod
    def load_config(cls, path):
        config, _ = cls.preprocess_config_attribute(path)
        predictor_class = get_predictor_infos(config)
        return predictor_class.config_class(**config.get("model_config", {}))

    @classmethod
    def load_from_config(cls, config: Union[str, Path, dict, ModuleConfig]):
        config, config_path = cls.preprocess_config_attribute(config)
        predictor_class = get_predictor_infos(config)
        if config_path is None:
            logger.error("Missing model path info")
            logger.error(config)
        logger.info(
            "Loading %s from %s",
            predictor_class.PREDICTOR_NAME,
            config_path,
            group=PREDICTOR,
        )
        return predictor_class(model_path=config_path)

    @classmethod
    def build_from_config(cls, config: Union[str, Path, dict, ModuleConfig]):
        config, _ = cls.preprocess_config_attribute(config)
        predictor_class = get_predictor_infos(config)
        logger.info("Building new %s", predictor_class.PREDICTOR_NAME, group=PREDICTOR)
        return predictor_class(config=config)

    @classmethod
    def _get_config_from_path(cls, config_path: Path):
        if config_path.is_dir():
            config_path = config_path / "config.json"
        if not config_path.exists():
            exception = PredictorNotFound(
                message=f'No file or directory at "{config_path}"'
            )
            logger.error(exception, group=PREDICTOR)
            raise exception
        try:
            with config_path.open(encoding="utf-8") as conf_file:
                return json.load(conf_file)
        except json.JSONDecodeError as json_exception:
            exception = PredictorUnprocessable(
                message="The provided config_file" " could not be loaded as Json"
            )
            logger.error(exception, group=PREDICTOR)
            raise exception from json_exception
