"""util functions for list, files and data"""
from typing import List

import numpy as np

from prescyent.utils.logger import logger, DATASET


def split_array_with_ratios(array: List, ratio1: float, ratio2: float,
                            ratio3: float = None, shuffle: bool = True):
    if len(array) < 1:
        raise ValueError("Can't split an empty array")
    if not isinstance(array, np.ndarray):
        array = np.array(array)

    if shuffle:
        np_rng = np.random.default_rng(2)   # have a deterministic shuffle for reruns
        np_rng.shuffle(array)
    if len(array) < 2:
        logger.warning("Only the first array could contain data",
                       group=DATASET)
        return array, list(), list()
    if not ratio3:
        len1 = round(len(array) * ratio1)
        if len(array) - len1 == 0:
            len1 -= 1
        return array[:len1], array[len1:]

    if len(array) < 3:
        logger.warning("Only 2 firsts array could contain data",
                       group=DATASET)
        return array[0], array[1], list()
    len1 = round(len(array) * ratio1)
    len2 = round(len(array) * (ratio1 + ratio2))
    return array[:len1], array[len1:len2], array[len2:]
