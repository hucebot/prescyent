"""Class and methods for the SCC Dataset, generating smooth circling trajectories
https://docs.scipy.org/doc/scipy/tutorial/interpolate/smoothing_splines.html
"""
import random
from pathlib import Path
from typing import Union, Dict, List

import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.interpolate import splrep, BSpline
from tqdm import tqdm

from . import metadata
from prescyent.dataset.dataset import MotionDataset
from prescyent.dataset.datasets.synthetic_circle_clusters.config import DatasetConfig
from prescyent.dataset.trajectories.trajectories import Trajectories
from prescyent.dataset.trajectories.trajectory import Trajectory
from prescyent.utils.logger import logger, DATASET


class Dataset(MotionDataset):
    """Simple dataset generating n 2D circles"""

    DATASET_NAME = "SCC"

    def __init__(
        self,
        config: Union[Dict, DatasetConfig, str, Path] = None,
        load_data_at_init: bool = False,
    ) -> None:
        logger.getChild(DATASET).info(
            f"Initializing {self.DATASET_NAME} Dataset",
        )
        self._init_from_config(config, DatasetConfig)
        super().__init__(name=self.DATASET_NAME, load_data_at_init=load_data_at_init)

    def prepare_data(self):
        """create a list of Trajectories from config variables"""
        train_trajectories = [
            self.generate_traj(i)
            for i in tqdm(range(int(self.config.num_traj * self.config.ratio_train)))
        ]
        logger.getChild(DATASET).info(
            f"Generated {len(train_trajectories)} train trajectories",
        )
        test_trajectories = [
            self.generate_traj(i)
            for i in tqdm(range(int(self.config.num_traj * self.config.ratio_test)))
        ]
        logger.getChild(DATASET).info(
            f"Generated {len(test_trajectories)} test trajectories",
        )
        val_trajectories = [
            self.generate_traj(i)
            for i in tqdm(range(int(self.config.num_traj * self.config.ratio_val)))
        ]
        logger.getChild(DATASET).info(
            f"Generated {len(val_trajectories)} val trajectories",
        )
        self.trajectories = Trajectories(
            train_trajectories, test_trajectories, val_trajectories
        )

    def generate_traj(self, traj_id: int) -> Trajectory:
        """Generate a circular 2D trajectory using the parameters from the config

        Args:
            id (int): id used for trajectory name

        Returns:
            Trajectory: new circular trajectory
        """
        cluster_id = random.randint(0, self.config.num_clusters - 1)
        starting_x = self.config.starting_xs[cluster_id]
        starting_y = self.config.starting_ys[cluster_id]
        radius = self.config.radius[cluster_id]
        # generate noisy circle with few point and high variation
        angles, noise_x, noise_y = generate_noisy_circle(
            radius=random.uniform(
                radius - self.config.radius_eps, radius + self.config.radius_eps
            ),
            num_points=self.config.num_imperfection_points,
            imperfection_range=self.config.imperfection_range,
        )
        # smoothing the noisy circle using scipy as in:
        # https://docs.scipy.org/doc/scipy/tutorial/interpolate/smoothing_splines.html
        tck_x = splrep(angles, noise_x, s=0)
        tck_y = splrep(angles, noise_y, s=0)
        xnew = np.linspace(0, 2 * np.pi, self.config.num_points)
        circle_x = BSpline(*tck_x)(xnew)
        circle_y = BSpline(*tck_y)(xnew)
        # Offset X
        circle_x = torch.from_numpy((circle_x + starting_x).astype("float32"))
        # Offset Y
        circle_y = torch.from_numpy((circle_y + starting_y).astype("float32"))
        # Create corresponding Trajectory
        trajectory = torch.cat(
            (circle_x.unsqueeze(1).unsqueeze(1), circle_y.unsqueeze(1).unsqueeze(1)),
            dim=-1,
        )
        trajectory = trajectory[:: self.config.subsampling_step]
        return Trajectory(
            trajectory,
            int(metadata.DEFAULT_FREQ / self.config.subsampling_step),
            metadata.FEATURES,
            file_path=f"synthetic_circle_{traj_id}_cluster_{cluster_id}",
            title=f"synthetic_circle_{traj_id}_cluster_{cluster_id}",
            point_names=metadata.POINT_LABELS,
            point_parents=metadata.POINT_PARENTS,
        )

    def plot_trajs(self, list_trajs: List[Trajectory], title="SCC Trajectories"):
        plt.figure()
        for traj in list_trajs:
            plt.plot(traj.tensor[:, :, 0].numpy(), traj.tensor[:, :, 1].numpy())
        plt.axis("equal")
        plt.title(title)
        plt.show()

    def plot_traj(self, traj: Trajectory):
        plt.figure()
        plt.plot(traj.tensor[:, :, 0].numpy(), traj.tensor[:, :, 1].numpy())
        plt.axis("equal")
        plt.title(traj.title)
        plt.show()


def generate_noisy_circle(radius, num_points, imperfection_range):
    angles = np.linspace(0, 2 * np.pi, num_points)
    # Adding imperfections to the radius
    perturbations = np.random.uniform(
        -imperfection_range, imperfection_range, num_points
    )
    perturbed_radius = (radius + perturbations).astype("float32")
    # Generating x and y values
    perturbed_circle_x = perturbed_radius * np.cos(angles)
    perturbed_circle_y = perturbed_radius * np.sin(angles)
    # Make last point = first point to have a clean circle
    perturbed_circle_x[-1] = perturbed_circle_x[0]
    perturbed_circle_y[-1] = perturbed_circle_y[0]
    return angles, perturbed_circle_x, perturbed_circle_y
