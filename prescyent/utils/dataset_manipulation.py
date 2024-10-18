"""util functions for list, files and data, plus methods to instanciate H36M in order to reproduce benchmarks"""
from typing import Any, List, Tuple

import numpy as np
import torch

from prescyent.utils.logger import logger, DATASET


def split_array_with_ratios(
    array: List,
    ratio1: float,
    ratio2: float,
    ratio3: float = None,
    shuffle: bool = True,
) -> Tuple[List[Any], List[Any], List[Any]]:
    """split a list in three given ratios

    Args:
        array (List): the list to split
        ratio1 (float): first ratio
        ratio2 (float): second ration
        ratio3 (float, optional): third ratio. Defaults to None.
        shuffle (bool, optional): shuffle the original array if true. Defaults to True.

    Raises:
        ValueError: if array is empty

    Returns:
        Tuple[List[Any], List[Any], List[Any]]: the three arrays
    """

    if len(array) < 1:
        raise ValueError("Can't split an empty array")
    if not isinstance(array, np.ndarray):
        array = np.array(array)
    if shuffle:
        np_rng = np.random.default_rng(2)  # have a deterministic shuffle for reruns
        np_rng.shuffle(array)
    if len(array) < 2:
        logger.getChild(DATASET).warning("Only the first array could contain data")
        return array, list(), list()
    if not ratio3:
        len1 = round(len(array) * ratio1)
        if len(array) - len1 == 0:
            len1 -= 1
        return array[:len1], array[len1:], list()
    if len(array) < 3:
        logger.getChild(DATASET).warning("Only 2 firsts array could contain data")
        return array[0], array[1], list()
    len1 = round(len(array) * ratio1)
    len2 = round(len(array) * (ratio1 + ratio2))
    return array[:len1], array[len1:len2], array[len2:]


def update_parent_ids(kept_indexes: List[int], parents: List[int]) -> List[int]:
    """update a reference map with a new list of indexes

    Args:
        kept_indexes (List[int]): the list of the ids that are kept
        parents (List[int]): reference map to update

    Returns:
        List[int]: updated reference map with new ids and -1 for missing ids
    """

    key_map = {k: v for v, k in enumerate(kept_indexes)}
    return [key_map.get(parents[i], -1) for i in kept_indexes]


def fkl_torch(rotmat, parent, offset, rotInd, expmapInd):
    """
    pytorch version of fkl.
    convert joint angles to joint locations
    batch pytorch version of the fkl() method
    :param angles: N*99
    :param parent:
    :param offset:
    :param rotInd:
    :param expmapInd:
    :return: N*joint_n*3
    """
    n = rotmat.data.shape[0]
    j_n = offset.shape[0]
    p3d = (
        torch.from_numpy(offset)
        .float()
        .to(rotmat.device)
        .unsqueeze(0)
        .repeat(n, 1, 1)
        .clone()
    )
    R = rotmat.reshape(n, j_n, 3, 3)
    for i in np.arange(1, j_n):
        if parent[i] >= 0:
            R[:, i, :, :] = torch.matmul(R[:, i, :, :], R[:, parent[i], :, :]).clone()
            p3d[:, i, :] = (
                torch.matmul(p3d[0, i, :], R[:, parent[i], :, :]) + p3d[:, parent[i], :]
            )
    return p3d, R


def expmap2quat_torch(exp):
    """
    Converts expmap to quaternion
    batch pytorch version ported from the corresponding numpy method above
    :param R: N*3
    :return: N*4
    """
    theta = torch.norm(exp, p=2, dim=1).unsqueeze(1)
    v = torch.div(exp, theta.repeat(1, 3) + 0.0000001)
    sinhalf = torch.sin(theta / 2)
    coshalf = torch.cos(theta / 2)
    q1 = torch.mul(v, sinhalf.repeat(1, 3))
    q = torch.cat((coshalf, q1), dim=1)
    return q


def expmap2rotmat_torch(r):
    """
    Converts expmap matrix to rotation
    batch pytorch version ported from the corresponding method above
    :param r: N*3
    :return: N*3*3
    """
    theta = torch.norm(r, 2, 1)
    r0 = torch.div(r, theta.unsqueeze(1).repeat(1, 3) + 0.0000001)
    r1 = torch.zeros_like(r0).repeat(1, 3)
    r1[:, 1] = -r0[:, 2]
    r1[:, 2] = r0[:, 1]
    r1[:, 5] = -r0[:, 0]
    r1 = r1.view(-1, 3, 3)
    r1 = r1 - r1.transpose(1, 2)
    n = r1.data.shape[0]
    R = (
        torch.eye(3, 3).repeat(n, 1, 1).float().to(r.device)
        + torch.mul(torch.sin(theta).unsqueeze(1).repeat(1, 9).view(-1, 3, 3), r1)
        + torch.mul(
            (1 - torch.cos(theta).unsqueeze(1).repeat(1, 9).view(-1, 3, 3)),
            torch.matmul(r1, r1),
        )
    )
    return R


def rotmat2xyz_torch(rotmat, _get_metadata_fn):
    """
    convert expmaps to joint locations
    :param rotmat: N*32*3*3
    :return: N*32*3
    """
    assert rotmat.shape[1] == 32
    parent, offset, rotInd, expmapInd = _get_metadata_fn()
    xyz, R = fkl_torch(rotmat, parent, offset, rotInd, expmapInd)
    return xyz, R


def find_indices_256(frame_num1, frame_num2, seq_len, input_n=10):
    """
    Adapted from
    https://github.com/una-dinosauria/human-motion-prediction/blob/master/src/seq2seq_model.py#L478
    In order to find the same action indices as in SRNN.
    """

    # Used a fixed dummy seed, following
    # https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/forecastTrajectories.py#L29
    SEED = 1234567890
    rng = np.random.RandomState(SEED)

    T1 = frame_num1 - 150
    T2 = frame_num2 - 150  # seq_len
    idxo1 = None
    idxo2 = None
    for _ in np.arange(0, 128):
        idx_ran1 = rng.randint(16, T1)
        idx_ran2 = rng.randint(16, T2)
        idxs1 = np.arange(idx_ran1 + 50 - input_n, idx_ran1 + 50 - input_n + seq_len)
        idxs2 = np.arange(idx_ran2 + 50 - input_n, idx_ran2 + 50 - input_n + seq_len)
        if idxo1 is None:
            idxo1 = idxs1
            idxo2 = idxs2
        else:
            idxo1 = np.vstack((idxo1, idxs1))
            idxo2 = np.vstack((idxo2, idxs2))
    return idxo1, idxo2


def find_indices_srnn(frame_num1, frame_num2, seq_len, input_n=10):
    """
    Adapted from
    https://github.com/una-dinosauria/human-motion-prediction/blob/master/src/seq2seq_model.py#L478
    In order to find the same action indices as in SRNN.
    """
    # Used a fixed dummy seed, following
    # https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/forecastTrajectories.py#L29
    SEED = 1234567890
    rng = np.random.RandomState(SEED)

    T1 = frame_num1 - 150
    T2 = frame_num2 - 150  # seq_len
    idxo1 = None
    idxo2 = None
    for _ in np.arange(0, 4):
        idx_ran1 = rng.randint(16, T1)
        idx_ran2 = rng.randint(16, T2)
        idxs1 = np.arange(idx_ran1 + 50 - input_n, idx_ran1 + 50 - input_n + seq_len)
        idxs2 = np.arange(idx_ran2 + 50 - input_n, idx_ran2 + 50 - input_n + seq_len)
        if idxo1 is None:
            idxo1 = idxs1
            idxo2 = idxs2
        else:
            idxo1 = np.vstack((idxo1, idxs1))
            idxo2 = np.vstack((idxo2, idxs2))
    return idxo1, idxo2
