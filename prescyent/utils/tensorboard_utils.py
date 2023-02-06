"""Util functions for tensorboard"""
import re
from typing import Tuple


def retreive_log_version_from_path(model_path: str) -> Tuple[str, int]:
    """use regexes to retrieve Tensorboard versioning

    Args:
        model_path (str): path to the model

    Returns:
        Tuple[str|None, str|None, int|None]: tuple of the logger's
            root_path, name and version for tensorboard versioning
            If not found, return a tuple of Nones
    """
    name, version = None, None
    TENSORBOARD_VERSION_REGEX = r"(.*?)([^\/]*?)[\/]version_(\d*)"
    match = re.search(TENSORBOARD_VERSION_REGEX, model_path)
    if match:
        model_path = match.groups()[0]
        name = match.groups()[1]
        version = int(match.groups()[2])
    return model_path, name, version
