"""Define project loggers in sub groups"""

import logging


PRESCYENT = "prescyent"
TRAINING = "training"
TESTING = "testing"
EVAL = "eval"
DATASET = "dataset"
PREDICTOR = "predictor"

LOG_GROUPS = [PRESCYENT, TRAINING, TESTING, EVAL, DATASET, PREDICTOR]


class OneLineExceptionFormatter(logging.Formatter):
    def formatException(self, exc_info):
        result = super().formatException(exc_info)
        return repr(result)

    def format(self, record):
        result = super().format(record)
        if record.exc_text:
            result = result.replace("\n", "")
        return result


def init_group_map(level):
    group_map = dict()
    for group_name in LOG_GROUPS:
        handler = logging.StreamHandler()
        formatter = OneLineExceptionFormatter(logging.BASIC_FORMAT)
        handler.setFormatter(formatter)
        _logger = logging.getLogger(str(group_name))
        _logger.setLevel(level)
        _logger.addHandler(handler)
        group_map[str(group_name)] = _logger
    return group_map


class GroupLogger():
    """
    """
    def __init__(self, level) -> None:
        self.group_map = init_group_map(level)

    def _set_group_level(self, group_name, level):
        """
        Set the logging level of the logger with given groupname.  level must be an int or a str.
        """
        self.group_map[str(group_name)].setLevel(level)

    def _set_group_levels(self, level):
        for group_name in LOG_GROUPS:
            self.group_map[str(group_name)].setLevel(level)

    def debug(self, msg: object, *args, group=PRESCYENT, **kwargs) -> None:
        return self.group_map[str(group)].debug(msg, *args, **kwargs)

    def info(self, msg: object, *args, group=PRESCYENT, **kwargs) -> None:
        return self.group_map[str(group)].info(msg, *args, **kwargs)

    def warning(self, msg: object, *args, group=PRESCYENT, **kwargs) -> None:
        return self.group_map[str(group)].warning(msg, *args, **kwargs)

    def error(self, msg: object, *args, group=PRESCYENT, **kwargs) -> None:
        return self.group_map[str(group)].error(msg, *args, **kwargs)

    def critical(self, msg: object, *args, group=PRESCYENT, **kwargs) -> None:
        return self.group_map[str(group)].critical(msg, *args, **kwargs)


logger = GroupLogger(level=logging.DEBUG)


def set_group_levels(level):
    logger._set_group_levels(level)


def set_group_level(group, level):
    logger._set_group_level(group, level)
