"""Module for trajectories classes"""
import csv
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
from scipy.spatial.transform import Rotation as ScipyRotation
from torch.serialization import MAP_LOCATION

from prescyent.utils.dataset_manipulation import update_parent_ids
from prescyent.utils.interpolate import update_tensor_frequency
from prescyent.utils.logger import logger, DATASET
from prescyent.dataset.features import (
    Feature,
    Features,
    Any,
    convert_tensor_features_to,
    get_distance,
    Rotation,
    RotationQuat,
)
from prescyent.dataset.features.rotation_methods import convert_to_quat


class Trajectory:
    """
    A trajectory represents a full dataset sample, that we can retrieve with its file name
    A trajectory tracks n features of m points over time, represented in a tensor of shape (frames, points, features)
    We describe its tensor with Features, a list of the Feature in the tensor.
    We also use point_parents and point_names to describe the entities we are tracking at each frame, and their relations.
    Finally we'll use context as additional data that we may want to pass to predictors along the trajectory
    """

    tensor: torch.Tensor
    """The tensor with the data describing the trajectory"""
    frequency: int
    """Frequency of the trajectory in Hz"""
    tensor_features: Features
    """Description of the features of the tensor with corresponding ids"""
    context: Dict[str, torch.Tensor]
    """Additionnal data about the trajectory. Allows some flexibility over the inputs and the way it will be passed to predictors.
    The tensors must have the same frequency as the base tensor"""
    file_path: str
    """Path to the file from which the trajectory is created"""
    title: str
    """Name given to the trajectory to describe it (especially in plots)"""
    point_parents: List[int]
    """List with the ids of the parent of each points, used to draw bones. -1 if no parent."""
    point_names: List[str]
    """List of a label to give to each point"""

    def __init__(
        self,
        tensor: torch.Tensor,
        frequency: float,
        tensor_features: Features = None,
        context: Dict[str, torch.Tensor] = {},
        file_path: Optional[str] = None,
        title: str = "trajectory_name",
        point_parents: List[int] = None,
        point_names: List[str] = None,
    ) -> None:
        """
        Args:
            tensor (torch.Tensor): The tensor with the data
            frequency (float): Frequency of the trajectory in Hz
            tensor_features (Features, optional): Description of the features of the tensor with corresponding ids. Defaults to None.
            context (Dict[str, torch.Tensor], optional): Additionnal data about the trajectory.
                    Allows some flexibility over the inputs and the way it will be passed to predictors
                    The tensors must have the same frequency as the base tensor. Default's to None
            file_path (str, optional): Path to the file from which the trajectory is created. Defaults to "trajectory_file_path".
            title (str, optional): Name given to the trajectory to describe it (especially in plots. Defaults to "trajectory_name".
            point_parents (List[int], optional): List with the ids of the parent of each points, used to draw bones. -1 if no parent. Defaults to None.
            point_names (List[str], optional): List of a label to give to each point. Defaults to None.
        """
        self.tensor = tensor
        if context is None:
            context = {}
        if context:
            # All context tensors must have the same frequency as the base tensor and a shape like (num_frames, feat_dim). For context, we consider we have one dict entry per "point"
            assert all(
                [c_tensor.shape[0] == tensor.shape[0] for c_tensor in context.values()]
            )
        self.context = context
        self.frequency = frequency
        self.file_path = file_path
        self.title = title
        if not point_parents:
            point_parents = [
                -1 for i in range(tensor.shape[1])
            ]  # default is -1 foreach point
        # All points are described
        assert len(point_parents) == tensor.shape[1]
        # All parents ids exists
        assert all(p_ids in range(-1, tensor.shape[1]) for p_ids in point_parents)
        self.point_parents = point_parents
        if not point_names:
            point_names = [
                f"point_{i}" for i in range(tensor.shape[1])
            ]  # default is "point_{i}" foreach point
        # All points are described
        assert len(point_names) == tensor.shape[1]
        self.point_names = point_names
        if tensor_features is None:
            tensor_features = Features([Any(list(range(tensor.shape[-1])))])
        elif isinstance(tensor_features, Feature):
            tensor_features = Features([tensor_features])
        # No duplicate ids in features
        assert len(set(tensor_features.ids)) == len(tensor_features.ids)
        # All feats are described
        assert len(tensor_features.ids) == tensor.shape[-1]
        self.tensor_features = tensor_features

    def __getitem__(self, index) -> torch.Tensor:
        return self.tensor[index]

    def __len__(self) -> int:
        return len(self.tensor)

    def __str__(self) -> str:
        return self.title

    @property
    def shape(self) -> torch.Size:
        """Tensor shape"""
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

    @property
    def context_len(self) -> int:
        """number of elements in the context"""
        return len(self.context)

    @property
    def context_dims(self) -> int:
        """sum of the dimensions of each element in context"""
        return sum([c_tensor.shape[-1] for c_tensor in self.context.values()])

    def convert_tensor_features(self, new_tensor_feats: Features):
        """convert trajectory's tensor to new requested features if possible,
            else raises an AttributeError
            self.tensor and self.tensor_features are updated.
        Args:
            new_tensor_feats (Features): new list of Feature
        """
        self.tensor = convert_tensor_features_to(
            self.tensor, self.tensor_features, new_tensor_feats
        )
        self.tensor_features = new_tensor_feats

    def update_frequency(self, target_freq: int) -> None:
        self.tensor, self.context = update_tensor_frequency(
            self.tensor, self.frequency, target_freq, self.tensor_features, self.context
        )
        self.frequency = target_freq

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

    def create_subtraj(
        self, points: List[int] = None, features: Features = None, context_keys=None
    ):
        """Create a subset of this trajectory with given new list of points and features

        Args:
            points (List[int]): ids of the points to keep. If None, the values will be same as self.
            features (Features): features to keep. If None, the values will be same as self.

        Returns:
            Trajectory: A new subtrajectory of self
        """
        if points is None:
            points = list(range(len(self.point_names)))
        if features is None:
            features = self.tensor_features
        if context_keys is None:
            context_keys = []
        context = {c_key: self.context[c_key] for c_key in context_keys}
        subtraj = Trajectory(
            tensor=self.tensor[:, points],
            tensor_features=self.tensor_features,
            frequency=self.frequency,
            context=context,
            file_path=self.file_path,
            title=self.title,
            point_parents=update_parent_ids(points, self.point_parents),
            point_names=[self.point_names[i] for i in points],
        )
        subtraj.convert_tensor_features(features)
        return subtraj

    @staticmethod
    def from_pt(
        pt_path: str,
        frequency: float,
        device: MAP_LOCATION = torch.device("cpu"),
        tensor_features: Features = None,
        context: Dict[str, torch.Tensor] = {},
        point_parents: List[int] = None,
        point_names: List[str] = None,
    ) -> object:
        """load a trajectory from a torch .pt file

        Args:
            pt_path (str): path of the saved tensor
            frequency (float): frequency of the trajectory
            device (MAP_LOCATION, optional): device where to load the tensor. Defaults to torch.device("cpu").
            tensor_features (Features, optional): Description of the features of the tensor with corresponding ids. Defaults to None.
            context (Dict[str, torch.Tensor], optional): Additionnal data about the trajectory.
                    Allows some flexibility over the inputs and the way it will be passed to predictors
                    The tensors must have the same frequency as the base tensor. Default's to None
            point_parents (List[int], optional): List with the ids of the parent of each points, used to draw bones. -1 if no parent. Defaults to None.
            point_names (List[str], optional): List of a label to give to each point. Defaults to None.

        Returns:
            Trajectory: _description_
        """
        tensor = torch.load(pt_path, map_location=device)
        return Trajectory(
            tensor,
            frequency=frequency,
            tensor_features=tensor_features,
            context=context,
            file_path=pt_path,
            title="_".join((Path(pt_path).parts)),
            point_parents=point_parents,
            point_names=point_names,
        )

    def to_pt(self, pt_path: str):
        """save the trajectory's tensor to .pt file

        Args:
            pt_path (str): path where to save the tensor
        """
        torch.save(self.tensor, pt_path)
