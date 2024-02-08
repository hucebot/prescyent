"""Module for trajectories classes"""
import csv
from pathlib import Path
from typing import List

from scipy.spatial.transform import Rotation as ScipyRotation
import torch

from prescyent.utils.interpolate import interpolate_trajectory_tensor_with_ratio
from prescyent.utils.logger import logger, DATASET
from prescyent.dataset.features import (
    Feature,
    Any,
    convert_tensor_features_to,
    Rotation,
    RotationQuat,
)
from prescyent.dataset.features.rotation_methods import convert_to_quat


class Trajectory:
    """
    An trajectory represents a full dataset sample, that we can retrieve with its file name
    An trajectory tracks n dimensions in time, represented in a tensor of shape (seq_len, n_dim)
    """

    tensor: torch.Tensor
    frequency: int
    tensor_features: List[Feature]
    file_path: str
    title: str
    point_parents: List[int]
    point_names: List[str]

    def __init__(
        self,
        tensor: torch.Tensor,
        frequency: int,
        tensor_features: List[Feature] = None,
        file_path: str = "trajectory_file_path",
        title: str = "trajectory_name",
        point_parents: List[int] = None,
        point_names: List[str] = None,
    ) -> None:
        self.tensor = tensor
        self.frequency = frequency
        self.file_path = file_path
        self.title = title
        if not point_parents:
            point_parents = [
                -1 for i in range(tensor.shape[1])
            ]  # default is -1 foreach point
        self.point_parents = point_parents
        if not point_names:
            point_names = [
                f"point_{i}" for i in range(tensor.shape[1])
            ]  # default is -1 foreach point
        self.point_names = point_names
        if tensor_features is None:
            tensor_features = [Any(list(range(tensor.shape[-1])))]
        elif isinstance(tensor_features, Feature):
            tensor_features = [tensor_features]
        self.tensor_features = tensor_features

    def __getitem__(self, index) -> torch.Tensor:
        return self.tensor[index]

    def __len__(self) -> int:
        return len(self.tensor)

    def __str__(self) -> str:
        return self.title

    @property
    def shape(self) -> torch.Size:
        return self.tensor.shape

    @property
    def duration(self) -> float:
        """duration in seconds"""
        return len(self.tensor) / self.frequency

    @property
    def num_points(self) -> int:
        """number of points in the trajectory"""
        return self.tensor.shape[1]

    @property
    def num_dims(self) -> int:
        """number of dimensions of each point in the trajectory"""
        return self.tensor.shape[2]

    def plot(self) -> None:
        # todo
        pass

    def vizualize_3d(self) -> None:
        # todo
        pass

    def compare(self, other) -> None:
        # todo
        pass

    def convert_tensor_features(self, new_tensor_feats: List[Feature]):
        self.tensor = convert_tensor_features_to(
            self.tensor, self.tensor_features, new_tensor_feats
        )
        self.tensor_features = new_tensor_feats

    def augment_frequency(self, augmentation_ratio: int) -> None:
        self.frequency = self.frequency * augmentation_ratio
        self.tensor = interpolate_trajectory_tensor_with_ratio(
            self.tensor, augmentation_ratio
        )

    def dump(
        self,
        output_path: str = None,
        output_format: str = "csv",
        write_header: bool = True,
    ) -> None:
        SUPPORTED_FORMATS = ["tsv", "csv"]
        if output_path is None:
            output_path = Path("data") / "dump" / f"{str(self)}.{output_format}"
            if not output_path.parent.exists():
                output_path.parent.mkdir(parents=True)
        if output_format == "tsv":
            delimiter = "\t"
        elif output_format == "csv":
            delimiter = ","
        if output_format in SUPPORTED_FORMATS:
            with open(output_path, "w", encoding="utf-8") as f:
                writer = csv.writer(f, delimiter=delimiter)
                if write_header:
                    header = self._get_header()
                    writer.writerow(header)
                writer.writerows(self._get_data())
            logger.getChild(DATASET).info(f"saved trajectory at {output_path}")
        else:
            raise NotImplementedError(
                f'output format "{output_format}" is not supported.'
                " Accepted values are {SUPPORTED_FORMATS}"
            )

    def _get_header(self) -> List[str]:
        return [
            f"{self.point_names[p]}_{self.dim_names[d]}"
            for p in range(self.num_points)
            for d in range(self.num_dims)
        ]

    @property
    def dim_names(self) -> List[str]:
        feature_dim_names = {}
        for feature in self.tensor_features:
            for i, feat_name in enumerate(feature.dims_names):
                feature_dim_names[feature.ids[i]] = feat_name
        return list(dict(sorted(feature_dim_names.items())).values())

    def _get_data(self) -> List[List[str]]:
        return self.tensor.flatten(1, 2).tolist()

    def get_scipy_rotation(self, seq_id: int, point_id: int):
        tensor = self.tensor[seq_id, point_id].clone()
        for feat in self.tensor_features:
            if isinstance(feat, Rotation):
                if not isinstance(feat, RotationQuat):
                    return ScipyRotation.from_quat(convert_to_quat(tensor[feat.ids]))
                return ScipyRotation.from_quat(tensor[feat.ids].numpy())
        return None
