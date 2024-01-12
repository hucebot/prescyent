"""util functions for tensors"""
from typing import Iterable, List, Tuple, Union
import torch

from prescyent.utils.quaternion_manipulation import quaternion_to_rotmatrix
from prescyent.utils.rotation_6d import rep6d_to_rotmatrix


def cat_tensor_with_seq_idx(
    preds: Union[List[torch.Tensor], torch.Tensor], flatt_idx: int = -1
):
    if isinstance(preds, list):
        preds = torch.stack(preds, dim=0)
    # if we have a list of preds
    if len(preds.shape) == 4:
        # we flatten the prediction to the last output of each prediciton
        # (seq_len, input_size, num_points, num_dim) -> (seq_len, num_points, num_dim)
        cat_preds = torch.zeros(preds.shape[0], preds.shape[2], preds.shape[3])
        for j, pred in enumerate(preds):
            cat_preds[j] = pred[flatt_idx]
        preds = cat_preds
    return preds


def cat_list_with_seq_idx(preds: torch.Tensor, flatt_idx: int = -1):
    # we flatten the prediction to the last output of each prediciton
    # list[Tensor(future_size, feature_size)] of len == pred_len
    #   -> Tensor(pred_len, feature_size)
    return torch.cat(
        [preds[0][:flatt_idx]] + [pred[flatt_idx].unsqueeze(0) for pred in preds], dim=0
    )


def is_tensor_is_batched(iterable: Iterable):
    return isinstance(iterable, torch.Tensor) and len(iterable.shape) >= 4


def trajectory_tensor_get_dim_limits(
    tensor: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """get min max values for each dim of a trajectory shaped tensor
    for example with a tensor of shape (100, 5, 3),
    where dims are x,y,z, we get the following result:
        ([min(x), min(y), min(z)], [max(x), max(y), max(z)])
    """
    tensor = tensor.transpose(1, 2)
    min_t = torch.min(tensor, dim=2)
    min_t = min_t.values.transpose(0, 1)
    min_t = torch.min(min_t, dim=1)
    max_t = torch.max(tensor, dim=2)
    max_t = max_t.values.transpose(0, 1)
    max_t = torch.max(max_t, dim=1)
    return min_t.values, max_t.values


def reshape_position_tensor(
    tensor: torch.Tensor,
) -> torch.Tensor:
    """
    Takes a tensor of shape (B, S, P, D) or (S, P, D)
    With :
        B = Batch size
        S = Sequence size
        P = Number of points
        D = Number of dimensions/features
    Here we accept tensors with D = 9, with each dimension being:
        Coordinate(x, y, z) and rep6d
    Or D=12
        Coordinate(x, y, z) and rotmatrix
    Or D=7
        Coordinate(x, y, z) and quaternion(x,y,z,w)
    Or D=6
        Coordinate(x, y, z) and euler(Z,Y,X)
    We  splits them in two new shapes with a x,y,z and rotmatrix"""
    batched = True
    assert isinstance(tensor, torch.Tensor)
    if not is_tensor_is_batched(tensor):  # add a B shape if none
        batched = False
        tensor.unsqueeze(0)
    if tensor.shape[-1] == 12:  # assert that this is a rotation as a rotmatrix
        pos_tensor = tensor
    elif tensor.shape[-1] in [6, 7, 9]:
        if tensor.shape[-1] == 6:  # assert that this is a rotation as a euler
            convert_method = NotImplementedError
        if tensor.shape[-1] == 7:  # assert that this is a rotation as a quaternion
            convert_method = quaternion_to_rotmatrix
        else:  # assert that this is a rotation as a rep6d
            convert_method = rep6d_to_rotmatrix
        pos_tensor = torch.zeros(*tensor.shape[:-1], 12)
        for b, batch in enumerate(tensor):
            for s, sequence in enumerate(batch):
                for p, point in enumerate(sequence):
                    pos_tensor[b][s][p] = torch.cat(
                        (point[:3], convert_method(point[3:]).flatten())
                    )
    else:
        raise AttributeError(
            "please check your tensor shape and that it represent a 3d position"
        )
    if batched is False:  # remove a B shape if none in input_tensor
        pos_tensor.squeeze(0)
    return pos_tensor
