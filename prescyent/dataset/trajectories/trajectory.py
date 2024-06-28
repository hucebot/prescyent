"""Module for trajectories classes"""
import csv
from pathlib import Path
from typing import Dict, List, Optional

from scipy.spatial.transform import Rotation as ScipyRotation
import torch

from prescyent.utils.dataset_manipulation import update_parent_ids
from prescyent.utils.interpolate import interpolate_trajectory_tensor_with_ratio
from prescyent.utils.logger import logger, DATASET
from prescyent.dataset.features import (
    Feature,
    Any,
    convert_tensor_features_to,
    get_distance,
    Rotation,
    RotationQuat,
)
from prescyent.dataset.features.rotation_methods import convert_to_quat


class Trajectory:
    """
    An trajectory represents a full dataset sample, that we can retrieve with its file name
    An trajectory tracks n features of m points over time, represented in a tensor of shape (frames, points, features)
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
            ]  # default is "point_{i}" foreach point
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

    def convert_tensor_features(self, new_tensor_feats: List[Feature]):
        """convert trajectory's tensor to new requested features if possible,
            else raises an AttributeError
            self.tensor and self.tensor_features are updated.
        Args:
            new_tensor_feats (List[Feature]): new list of Feature
        """
        self.tensor = convert_tensor_features_to(
            self.tensor, self.tensor_features, new_tensor_feats
        )
        self.tensor_features = new_tensor_feats

    def augment_frequency(self, augmentation_ratio: int) -> None:
        """Augment the tensor's frequency using a ratio and linear interpolation
            new self.tensor will have shape (B, S*ratio, P, D)
            and, self.frequency is also updated this way
        Args:
            augmentation_ratio (int): ratio used for interpolation
        """
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
        """Outputs the trajectory tensors as a csv/tsv with a line per frame and one column per feature.

        Args:
            output_path (str, optional): filename used to create file. Defaults to None.
            output_format (str, optional): "cvs" or "tsv". Defaults to "csv".
            write_header (bool, optional): If true headers are included using trajectory's metadata. Defaults to True.

        Raises:
            NotImplementedError: if format isn't known
        """
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
        """returns list of headers for csv dump

        Returns:
            List[str]: list of columns names if data is dumped
        """
        return [
            f"{self.point_names[p]}_{self.dim_names[d]}"
            for p in range(self.num_points)
            for d in range(self.num_dims)
        ]

    @property
    def dim_names(self) -> List[str]:
        """returns names for the last tensor dim, based self.features

        Returns:
            List[str]: List of names
        """
        feature_dim_names = {}
        for feature in self.tensor_features:
            for i, feat_name in enumerate(feature.dims_names):
                feature_dim_names[feature.ids[i]] = feat_name
        return list(dict(sorted(feature_dim_names.items())).values())

    def _get_data(self) -> List[List[float]]:
        """tensor formater used for data dumps

        Returns:
            List[List[float]]: tensor infos as a list, with shape (seq, point * dim)
        """
        return self.tensor.flatten(1, 2).tolist()

    def get_scipy_rotation(self, seq_id: int, point_id: int):
        """Return scipy rotation for a given frame and point, if a rotation feat exists in self

        Args:
            seq_id (int): frame id in sequence
            point_id (int): point id

        Returns:
            scipy.spatial.transform.Rotation: Corresponding scipy rotation
        """
        tensor = self.tensor[seq_id, point_id].clone()
        for feat in self.tensor_features:
            if isinstance(feat, Rotation):
                if not isinstance(feat, RotationQuat):
                    return ScipyRotation.from_quat(
                        convert_to_quat(tensor[feat.ids].unsqueeze(0))
                    )
                return ScipyRotation.from_quat(tensor[feat.ids].numpy())
        return None

    def compare_with(
        self, trajectories: List[object], offsets: Optional[List[int]] = None
    ) -> List[Dict[str, float]]:
        """Return mean feature distance between self and each traj in Trajectories

        Args:
            trajectories (List[object]): List of Trajectory to compare self with
            offsets (Optional[List[int]], optional): id of the frame to start comparition from in self. If None then no offset, first frame is 0.

        Returns:
            List[Dict[str, float]]: List of the mean distances for each feat in the form {Feature.name: Tensor}
        """
        if offsets is None:
            offsets = [0 for _ in trajectories]
        metrics = []
        num_points = self.tensor.shape[1]
        assert all(
            [traj.tensor.shape[1] == num_points for traj in trajectories]
        )  # Plotted trajs must have number of points
        for traj, offset in zip(trajectories, offsets):
            truth_tensor = self.tensor[offset:]
            mean_dists = get_distance(
                truth_tensor,
                self.tensor_features,
                traj.tensor[: len(truth_tensor)],
                traj.tensor_features,
                get_mean=True,
            )
            metrics.append(mean_dists)
        return metrics

    def create_subtraj(self, points: List[int] = None, features: List[Feature] = None):
        """Create a subset of this trajectory with given new list of points and features

        Args:
            points (List[int]): ids of the points to keep. If None, the values will be same as self.
            features (List[Feature]): features to keep. If None, the values will be same as self.

        Returns:
            Trajectory: A new subtrajectory of self
        """
        if points is None:
            points = list(range(self.point_names))
        if features is None:
            features = self.tensor_features
        subtraj = Trajectory(
            tensor=self.tensor[:, points],
            tensor_features=self.tensor_features,
            frequency=self.frequency,
            file_path=self.file_path,
            title=self.title,
            point_parents=update_parent_ids(points, self.point_parents),
            point_names=[self.point_names[i] for i in points],
        )
        subtraj.convert_tensor_features(features)
        return subtraj
