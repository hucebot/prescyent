"""Class and methods for the AndyDataset Dataset
https://andydataset.loria.fr/
"""
from pathlib import Path
from typing import List, Union, Dict
import xml.etree.ElementTree as ET

import torch
from tqdm import tqdm

from prescyent.utils.logger import logger, DATASET
from prescyent.utils.dataset_manipulation import (
    split_array_with_ratios,
    update_parent_ids,
)
import prescyent.dataset.datasets.andydataset.metadata as metadata
from prescyent.dataset.datasets.andydataset.config import DatasetConfig
from prescyent.dataset.features import Coordinate
from prescyent.dataset.trajectories.trajectories import Trajectories
from prescyent.dataset.trajectories.trajectory import Trajectory
from prescyent.dataset.dataset import MotionDataset


class Dataset(MotionDataset):
    """
    https://andydataset.loria.fr/
    Dataset is not splitted into test / train / val
    It as to be at initialisation, through the parameters
    """

    DATASET_NAME = "AndyDataset"

    def __init__(
        self,
        config: Union[Dict, DatasetConfig, str, Path] = None,
        load_data_at_init: bool = True,
    ) -> None:
        self._init_from_config(config, DatasetConfig)
        super().__init__(name=self.DATASET_NAME, load_data_at_init=load_data_at_init)

    def prepare_data(self):
        """get trajectories from files or web"""
        if not Path(self.config.data_path).exists():
            logger.getChild(DATASET).warning(
                "Dataset files not found at path %s",
                self.config.data_path,
            )
            self._get_from_web()
        self.trajectories = self._load_files()

    def _get_from_web(self) -> None:
        raise NotImplementedError(
            "This dataset must be downloaded manually, "
            "please follow the instructions in the README"
        )

    # load a set of trajectory, keeping them separate
    def _load_files(self) -> Trajectories:
        logger.getChild(DATASET).debug(
            "Searching Dataset files from path %s", self.config.data_path
        )
        if self.config.use_pt:
            files = list(Path(self.config.data_path).rglob(self.config.pt_glob_dir))
            if len(files) == 0:
                self.config.use_pt = False
                files = list(Path(self.config.data_path).rglob(self.config.glob_dir))
        else:
            files = list(Path(self.config.data_path).rglob(self.config.glob_dir))
        files.sort()
        if len(files) == 0:
            logger.getChild(DATASET).error(
                "No files matching '%s' rule for this path %s",
                self.config.glob_dir,
                self.config.data_path,
            )
            raise FileNotFoundError(self.config.data_path)
        train_files, test_files, val_files = split_array_with_ratios(
            files,
            self.config.ratio_train,
            self.config.ratio_test,
            self.config.ratio_val,
            shuffle=self.config.shuffle_data_files,
        )
        train = self.pathfiles_to_trajectories(train_files)
        logger.getChild(DATASET).info(
            "Created %d trajectories in the train set", len(train)
        )
        test = self.pathfiles_to_trajectories(test_files)
        logger.getChild(DATASET).info(
            "Created %d trajectories in the test set", len(test)
        )
        val = self.pathfiles_to_trajectories(val_files)
        logger.getChild(DATASET).info(
            "Created %d trajectories in the val set", len(val)
        )
        return Trajectories(train, test, val)

    def pathfiles_to_trajectories(self, files: List) -> List[Trajectory]:
        """util method to turn a list of pathfiles to a list of Trajectories
        :param files: list of files
        :type files: List
        :param delimiter: delimiter to split the data on, defaults to ','
        :type delimiter: str, optional
        :raises FileNotFoundError: _description_
        :return: the data of the dataset, grouped per file
        :rtype: list
        """
        used_joints = self.config.used_joints
        trajectory_arrray = list()
        for file_path in tqdm(files):
            # If we don't have pt or don't want to use them, we parse the whole xml
            if not self.config.use_pt:
                tree = ET.parse(file_path)
                root = tree.getroot()
                subject = root.find(f"./{metadata.SCHEMA}subject")
                frames = subject.findall(
                    f'./{metadata.SCHEMA}frames/{metadata.SCHEMA}frame[@type="normal"]'
                )
                frame_list = []
                for frame in frames:
                    # for frame in frames:
                    position = torch.FloatTensor(
                        [
                            float(x)
                            for x in frame.find(
                                f"./{metadata.SCHEMA}position"
                            ).text.split()
                        ]
                    ).reshape(23, 3)
                    orientation = torch.FloatTensor(
                        [
                            float(x)
                            for x in frame.find(
                                f"./{metadata.SCHEMA}orientation"
                            ).text.split()
                        ]
                    ).reshape(23, 4)
                    # (w, x, y, z) => (x, y, z, w)
                    orientation = orientation[:, [1, 2, 3, 0]]
                    # We ensure we have the quaternion with a positive w
                    indices = torch.nonzero(orientation[:, -1] < 0)
                    orientation[indices] = -orientation[indices]
                    frame_tensor = torch.cat((position, orientation), dim=-1)
                    frame_list.append(frame_tensor.tolist())
                torch_tensor = torch.FloatTensor(frame_list)
                torch.save(torch_tensor, f"{file_path}.pt")
            # Else we load directly the saved torch representation of the base trajectory
            else:
                torch_tensor = torch.load(file_path)
            if self.config.make_joints_position_relative_to is not None:
                for feat in metadata.FEATURES:
                    if isinstance(feat, Coordinate):
                        torch_tensor[:, :, feat.ids] -= torch.index_select(
                            torch_tensor,
                            1,
                            torch.IntTensor(
                                [self.config.make_joints_position_relative_to]
                            ),
                        )[:, :, feat.ids]
            torch_tensor = torch_tensor[
                :, used_joints
            ]  # Keep only joints_ids from config
            point_parents = update_parent_ids(used_joints, metadata.SEGMENT_PARENTS)
            point_names = [metadata.SEGMENT_LABELS[i] for i in used_joints]
            traj = Trajectory(
                torch_tensor,
                frequency=int(metadata.BASE_FREQUENCY),
                tensor_features=metadata.FEATURES,
                file_path=file_path,
                title=Path(file_path).name.split(".")[0],
                point_parents=point_parents,
                point_names=point_names,
            )
            trajectory_arrray.append(traj)
        return trajectory_arrray
