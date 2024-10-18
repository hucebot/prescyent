"""Define project loggers in sub groups"""

import logging
from typing import Union


PRESCYENT = "prescyent"
TRAINING = "training"
TESTING = "testing"
EVAL = "eval"
DATASET = "dataset"
PREDICTOR = "predictor"

LOG_GROUPS = [PRESCYENT, TRAINING, TESTING, EVAL, DATASET, PREDICTOR]


class OneLineExceptionFormatter(logging.Formatter):
    """converts logged exceptions to oneline printable for more readable logging"""

    def formatException(self, exc_info):
        result = super().formatException(exc_info)
        return repr(result)

    def format(self, record):
        result = super().format(record)
        if record.exc_text:
            result = result.replace("\n", "")
        return result


def init_logger(level: Union[int, str]) -> logging.Logger:
    """initialise lib's logger oin given level with default handler and formatter

    Args:
        level (Union[int, str]): log level

    Returns:
        Logger: Logger instance
    """

    handler = logging.StreamHandler()
    _format = "%(name)-10s : %(levelname)-5s - %(message)s [%(pathname)s:%(lineno)d]"
    formatter = OneLineExceptionFormatter(_format)
    handler.setFormatter(formatter)
    _logger = logging.getLogger("prescyent")
    _logger.setLevel(level)
    _logger.addHandler(handler)
    return _logger


logger = init_logger(level=logging.DEBUG)
