"""util functions for list, files and data"""
from prescyent.utils.logger import logger, DATASET
from typing import List

import numpy as np


def split_array_with_ratios(array: List, ratio1: float, ratio2: float,
                            ratio3: float = None, shuffle: bool = True):
    if len(array) < 1:
        raise ValueError("Can't split an empty array")
    if not isinstance(array, np.ndarray):
        array = np.array(array)

    if shuffle:
        np.random.shuffle(array)
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


def pathfiles_to_array(files: List,
                       delimiter: str = ',',
                       start: int = None,
                       end: int = None,
                       subsampling_step: int = 0,
                       dimensions: List[int] = None) -> list:
    """util method to turn a list of pathfiles to a list of their data

    :param files: list of files
    :type files: List
    :param delimiter: delimiter to split the data on, defaults to ','
    :type delimiter: str, optional
    :param subsampling_step: the step used for final list, allows to skip data, defaults to 0
    :type subsampling_step: int, optional
    :param dimensions: _description_, defaults to 0
    :type dimensions: List[int], optional, defaults to None
    :raises FileNotFoundError: _description_
    :return: the data of the dataset, grouped per file
    :rtype: list
    """
    if start is None:
        start = 0
    result_arrray = list()
    for file in files:
        if not file.exists():
            logger.error("file does not exist:", file,
                         group=DATASET)
            raise FileNotFoundError(file)
        file_array = np.loadtxt(file, delimiter=delimiter)
        if end is None:
            end = len(file_array)
        if dimensions is not None:
            result_arrray += [file_array[start:end:subsampling_step, dimensions]]
        else:
            result_arrray += [file_array[start:end:subsampling_step]]
    return result_arrray
