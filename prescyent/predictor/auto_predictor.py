import json
from pathlib import Path
from typing import Union

from prescyent.predictor.lightning.module_config import ModuleConfig
from prescyent.utils.logger import logger, PREDICTOR
from prescyent.predictor.lightning.sequence.linear import LinearPredictor
from prescyent.predictor.lightning.sequence.mlp import MlpPredictor
from prescyent.predictor.lightning.autoreg.sarlstm import SARLSTMPredictor
from prescyent.predictor.lightning.sequence.seq2seq import Seq2SeqPredictor

try:
    from prescyent.experimental.siMLPe import SiMLPePredictor
    use_experimental = True
except ModuleNotFoundError:
    use_experimental = False
    logger.warning("modules from experimental package will not be instanciable",
                group=PREDICTOR)


predictor_list = [LinearPredictor, SARLSTMPredictor,
                  Seq2SeqPredictor, MlpPredictor]
if use_experimental:
    predictor_list.append(SiMLPePredictor)


predictor_map = {p.PREDICTOR_NAME: p for p in predictor_list}


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
        predictor_class = predictor_map.get(predictor_class_name, None)
        if predictor_class is None:
            logger.error("Could not find a predictor class matching %s",
                         predictor_class_name, group=PREDICTOR)
            raise AttributeError(predictor_class_name)
        if config_path is None:
            logger.error("Missing model path info")
        logger.info("Loading %s from %s", predictor_class, config_path,
                    group=PREDICTOR)
        return predictor_class(model_path=config_path)

    @classmethod
    def build_from_config(cls, config: Union[str, Path, dict, ModuleConfig]):
        if isinstance(config, (str, Path)):
            config = cls._get_config_from_path(Path(config))
        if isinstance(config, ModuleConfig):
            config = config.dict()
        predictor_class_name = config.get("name", None)
        predictor_class = predictor_map.get(predictor_class_name, None)
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
        with config_path.open(encoding="utf-8") as conf_file:
            return json.load(conf_file)
