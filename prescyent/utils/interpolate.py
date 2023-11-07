from typing import Iterable

import numpy as np
import torch


def interpolate_iterable_with_ratio(input_list: Iterable, interpolation_ratio:int) -> Iterable:
    """output has size (len(input_list) - 1) * ratio + 1 """
    output_list = []
    for i, _ in enumerate(input_list[:-1]):
        x_interpolated = np.linspace(input_list[i], input_list[i+1], interpolation_ratio + 1)
        output_list += list(x_interpolated)[:-1]
    output_list.append(input_list[-1])
    return output_list

def interpolate_trajectory_tensor_with_ratio(input_tensor: torch.Tensor, interpolation_ratio: int) -> torch.Tensor:
    assert len(input_tensor.shape) == 3
    input_tensor = torch.transpose(input_tensor, 0, 1)
    input_tensor = torch.transpose(input_tensor, 1, 2)
    # for each dim and for each point we interpolate on sequence
    output = []
    for point_t in input_tensor:
        point = []
        for dim in point_t:
            point.append(interpolate_iterable_with_ratio(dim, interpolation_ratio))
        output.append(point)
    output_tensor = torch.FloatTensor(output)
    output_tensor = torch.transpose(output_tensor, 1, 2)
    output_tensor = torch.transpose(output_tensor, 0, 1)
    return output_tensor