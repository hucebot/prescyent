"""util functions for list, files and data"""

from typing import List

import numpy as np


def split_array_with_ratios(array: List, ratio1: float, ratio2: float,
                            ratio3: float = None, shuffle: bool=True):
    if len(array) < 1:
        raise ValueError("Can't split an empty array")
    if not isinstance(array, np.ndarray):
        array = np.array(array)
    print(len(array))

    if shuffle:
        np.random.shuffle(array)
    if len(array) < 2:
        print("WARNING: Only the first array could contain data")
        return array, list(), list()
    if not ratio3:
        len1 = round(len(array) * ratio1)
        if len(array) - len1 == 0:
            len1 -= 1
        return array[:len1], array[len1:]

    if len(array) < 3:
        print("WARNING: Only 2 firsts array could contain data")
        return array[0], array[1], list()
    len1 = round(len(array) * ratio1)
    len2 = round(len(array) * (ratio1 + ratio2))
    return array[:len1], array[len1:len2], array[len2:]


def pathfiles_to_array(files: List,
                       delimiter: str = ',',
                       start: int = None,
                       end: int = None,
                       skip_data: int = 0,
                       column: int = 0) -> list:
    """util method to turn a list of pathfiles to a list of their data

    :param files: list of files
    :type files: List
    :param delimiter: delimiter to split the data on, defaults to ','
    :type delimiter: str, optional
    :param skip_data: the step used for final list, allows to skip data, defaults to 0
    :type skip_data: int, optional
    :param column: _description_, defaults to 0
    :type column: int, optional
    :raises FileNotFoundError: _description_
    :return: the data of the dataset, grouped per file
    :rtype: list
    """
    if start is None:
        start = 0
    result_arrray = list()
    for file in files:
        if not file.exists():
            print("ERROR, file does not exist:", file)
            raise FileNotFoundError(file)
        file_array = np.loadtxt(file, delimiter=delimiter)
        if end is None:
            end = len(file_array)
        if column is not None:
            result_arrray += [file_array[start:end:skip_data, column]]
        else:
            result_arrray += [file_array[start:end:skip_data]]
    return result_arrray
