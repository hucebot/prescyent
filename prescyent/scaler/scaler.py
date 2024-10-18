from pathlib import Path
from typing import List, Optional, Union

try:
    import cPickle as pickle
except ImportError:
    import pickle

import torch
import torch.utils

from prescyent.dataset.features import Features, Rotation
from prescyent.utils.logger import logger, PREDICTOR
from prescyent.utils.enums import Scalers
from .config import ScalerConfig
from .scaling_methods.standardizer import Standardizer
from .scaling_methods.normalizer import Normalizer


scaler_map = {
    Scalers.STANDARDIZATION: Standardizer,
    Scalers.NORMALIZATION: Normalizer,
    None: None,
}


class Scaler:
    """Class to handle feature scaling.
    We deal with all feature wise operations on this class.
    And train and call feature scaling methods."""

    def __init__(self, config: ScalerConfig) -> None:
        self.config = config
        self.scalers = dict()
        self.status = "untrained"
        self.dataset_features = None

    def describe(self) -> str:
        """output string that describe the scaler's parameters"""

        if self.status == "untrained":
            return "Untrained Scaler"
        if not self.config.do_feature_wise_scaling:
            return f"Scaler with {self.scalers['feature'].__class__.__name__} on dim {self.scalers['feature'].dim}"
        _str = "Feature Wise Scaler with:"
        for feat_name, scaler in self.scalers.items():
            _str += "\n" + 5 * "    "
            _str += (
                f"{scaler.__class__.__name__} on dim {scaler.dim} for Feat {feat_name}"
            )
        return _str

    def train(
        self,
        dataset_dataloader: torch.utils.data.DataLoader,
        dataset_features: Features,
    ):
        """train scalers from a dataloader over dataset trajectories

        Args:
            dataset_dataloader (torch.utils.data.DataLoader): dataloader over the whole dataset's train trajectories
            dataset_features (Features): features of the dataset

        Raises:
            AttributeError: scaler not available
        """

        scaler = scaler_map.get(self.config.scaler, -1)
        if scaler == -1:
            raise AttributeError(
                f"Please choose a valid scaler in NormalizationConfig from the following: {scaler_map.keys()}"
            )
        if dataset_features is None or not self.config.do_feature_wise_scaling:
            if scaler is not None and self.config.scaling_axis is not None:
                self.scalers["feature"] = scaler(self.config.scaling_axis)
                self.scalers["feature"].train(dataset_dataloader)
        else:
            for feat in dataset_features:
                if (
                    not isinstance(feat, Rotation) or self.config.scale_rotations
                ):  # If its not a Rotation, or if we scale them
                    if scaler is not None and self.config.scaling_axis is not None:
                        self.scalers[feat.name] = scaler(self.config.scaling_axis)
                        self.scalers[feat.name].train(dataset_dataloader, feat.ids)
        self.status = "trained"
        self.dataset_features = dataset_features
        logger.getChild(PREDICTOR).info(f"Trained {self.describe()}")

    def scale(
        self,
        input_t: torch.Tensor,
        in_points_ids: Optional[List[int]] = None,
        features: Optional[Features] = None,
    ) -> torch.Tensor:
        """method to scale a tensor

        Args:
            input_t (torch.Tensor): traj tensor to scale
            in_points_ids (Optional[List[int]], optional): ids of the points to keeps in the traj tensor. Defaults to None.
            features (Optional[Features], optional): traj tensor's Features. Defaults to None.

        Raises:
            AssertionError: Scaler wasn't trained
            AttributeError: if tried to scale a rotation that wasn't in the trained dataset

        Returns:
            torch.Tensor: scaled tensor
        """

        if self.status == "untrained":
            raise AssertionError(
                "Scaler wasn't trained, please train before using method"
            )
        assert len(input_t.shape) == 4
        if not self.scalers:  # If we don't have any scaler init, return as is
            return input_t
        output_t = input_t.clone()
        if not self.config.do_feature_wise_scaling or features is None:
            return self.scalers["feature"].scale(output_t, in_points_ids)
        for feat in features:
            if not isinstance(feat, Rotation) or self.config.scale_rotations:
                feat_ids = None
                if feat not in self.dataset_features:  # If don't find the feature
                    if isinstance(
                        feat, Rotation
                    ):  # we cannot "convert" the feature for a rotation to another
                        raise AttributeError(
                            f"We cannot scale unknown rotation, know feature from dataset training are {self.dataset_features}"
                        )
                    if not self.scalers.get(
                        feat.name, None
                    ):  # we cannot "convert" the feature with a different name
                        logger.getChild(PREDICTOR).warning(
                            f"Scaler for feat {feat.name} wasn't trained, skipping this feat"
                        )
                        continue
                    feat_ids = feat.ids
                output_t[..., feat.ids] = self.scalers[feat.name].scale(
                    output_t[..., feat.ids], in_points_ids, feat_ids
                )
        return output_t

    def unscale(
        self,
        input_t: torch.Tensor,
        out_points_ids: Optional[List[int]] = None,
        features: Optional[Features] = None,
    ) -> torch.Tensor:
        """method to unscale a tensor

        Args:
            input_t (torch.Tensor): tensor to unscale
            out_points_ids (Optional[List[int]], optional): ids of the points to keeps in the traj tensor. Defaults to None.
            features (Optional[Features], optional): traj tensor's Features. Defaults to None.

        Raises:
            AssertionError: Scaler wasn't trained
            AttributeError: if tried to unscale a rotation that wasn't in the trained dataset

        Returns:
            torch.Tensor: unscaled tensor
        """

        if self.status == "untrained":
            raise AssertionError(
                "Scaler wasn't trained, please train before using method"
            )
        assert len(input_t.shape) == 4
        if not self.scalers:
            return input_t
        output_t = input_t.clone()
        if not self.config.do_feature_wise_scaling or features is None:
            return self.scalers["feature"].unscale(
                output_t, out_points_ids, features.ids
            )
        for feat in features:
            if not isinstance(feat, Rotation) or self.config.scale_rotations:
                feat_ids = None
                if feat not in self.dataset_features:
                    if isinstance(feat, Rotation):
                        raise AttributeError(
                            f"We cannot unscale unknown rotation, know feature from dataset training are {self.dataset_features}"
                        )
                    feat_ids = feat.ids
                output_t[..., feat.ids] = self.scalers[feat.name].unscale(
                    output_t[..., feat.ids], out_points_ids, feat_ids
                )
        return output_t

    def save(self, file_path: Union[str, Path]):
        """save the scaler to path using pickle

        Args:
            file_path (Union[str, Path]): path where to save the scaler
        """

        if isinstance(file_path, str):
            file_path = Path(file_path)
        if not file_path.parent.exists():
            file_path.parent.mkdir(parents=True)
        with file_path.open("wb") as file_wb:
            pickle.dump(self.__dict__, file_wb, 2)

    @classmethod
    def load(cls, file_path: Union[str, Path]):
        """load a scaler from path using pickle

        Args:
            file_path (Union[str, Path]): path where to the scaler
        """

        if isinstance(file_path, str):
            file_path = Path(file_path)
        with file_path.open("rb") as file_rb:
            dict_data = pickle.load(file_rb)
        scaler = cls(dict_data["config"])
        scaler.__dict__.clear()
        scaler.__dict__.update(dict_data)
        return scaler
