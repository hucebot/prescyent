"""Class and methods for the SCC Dataset, generating smooth circling trajectories
https://docs.scipy.org/doc/scipy/tutorial/interpolate/smoothing_splines.html
"""
from pathlib import Path
import tempfile
from typing import Tuple, Union, Dict, List

import h5py
import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.interpolate import splrep, BSpline
from tqdm.auto import tqdm

from prescyent.dataset.dataset import TrajectoriesDataset
from prescyent.dataset.hdf5_utils import write_metadata
from prescyent.dataset.trajectories.trajectories import Trajectories
from prescyent.dataset.trajectories.trajectory import Trajectory
from prescyent.utils.interpolate import update_tensor_frequency
from prescyent.utils.logger import logger, DATASET

from . import metadata
from .config import SCCDatasetConfig


class SCCDataset(TrajectoriesDataset):
    """Simple dataset generating n 2D circles"""

    DATASET_NAME = "SCC"

    def __init__(
        self,
        config: Union[Dict, SCCDatasetConfig, str, Path] = None,
    ) -> None:
        logger.getChild(DATASET).info(
            f"Initializing {self.DATASET_NAME} Dataset",
        )
        self._init_from_config(config, SCCDatasetConfig)
        super().__init__(name=self.DATASET_NAME)

    def prepare_data(self):
        """create a list of Trajectories from config variables"""
        if hasattr(self, "_trajectories"):
            return
        self.tmp_hdf5 = tempfile.NamedTemporaryFile(suffix=".hdf5")
        frequency = metadata.DEFAULT_FREQ
        tmp_hdf5_data = h5py.File(self.tmp_hdf5.name, "w")
        write_metadata(
            tmp_hdf5_data,
            frequency=frequency,
            point_parents=metadata.POINT_PARENTS,
            point_names=metadata.POINT_LABELS,
            features=metadata.DEFAULT_FEATURES,
        )
        np.random.seed(self.config.seed)
        traj_id = 0
        for c in tqdm(
            range(self.config.num_clusters),
            desc="Iterating on clusters",
            colour="green",
        ):
            cluster_counter = 0
            context = {}
            for cluster_counter in tqdm(
                range(self.config.num_trajs[c]),
                desc=f"Creating trajs in cluster {c}",
                colour="blue",
            ):
                tensor = self.generate_traj(c)
                tensor, context = update_tensor_frequency(
                    tensor,
                    frequency,
                    self.config.frequency,
                    metadata.DEFAULT_FEATURES,
                    context,
                )
                if cluster_counter < int(
                    self.config.num_trajs[c] * self.config.ratio_train
                ):
                    key = "train"
                elif cluster_counter < int(
                    self.config.num_trajs[c]
                    * (self.config.ratio_train + self.config.ratio_test)
                ):
                    key = "test"
                else:
                    key = "val"
                tmp_hdf5_data.create_dataset(
                    f"{key}/cluster_{c}/trajectory_{traj_id}/traj",
                    data=tensor,
                )
                traj_id += 1
        self.trajectories = Trajectories.__init_from_hdf5__(self.tmp_hdf5.name)
        tmp_hdf5_data.close()
        np.random.seed()

    def generate_traj(self, cluster_id: int) -> Trajectory:
        """Generate a circular 2D trajectory using the parameters from the config

        Args:
            id (int): id used for trajectory name

        Returns:
            Trajectory: new circular trajectory
        """
        starting_x = self.config.starting_xs[cluster_id]
        starting_y = self.config.starting_ys[cluster_id]
        radius = self.config.radius[cluster_id]
        # generate noisy circle with few point and high variation
        angles, noise_x, noise_y = generate_noisy_circle(
            radius=np.random.uniform(
                radius - self.config.radius_eps, radius + self.config.radius_eps
            ),
            num_points=self.config.num_perturbation_points,
            perturbation_range=self.config.perturbation_range,
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
        tensor = torch.cat(
            (circle_x.unsqueeze(1).unsqueeze(1), circle_y.unsqueeze(1).unsqueeze(1)),
            dim=-1,
        )
        return tensor

    def plot_trajs(
        self,
        list_trajs: List[Trajectory],
        title="SCC Trajectories",
        save_path: str = None,
        legend_labels: List[str] = None,
    ):
        """plot all trajectories as a 2D top view of x and y positions at all frames

        Args:
            list_trajs (List[Trajectory]): all trajectories to plot
            title (str, optional): title for the plot. Defaults to "SCC Trajectories".
            save_path (str, optional): where to save the plot. If None we do not save and use plt.show instead. Defaults to None.
            legend_labels (List[str], optional): strings used to legend the plot. Defaults to None.
        """
        from prescyent.evaluator.plotting import save_plot_and_close

        plt.figure()
        fig, ax = plt.subplots()
        for traj in list_trajs:
            ax.plot(traj.tensor[:, :, 0].numpy(), traj.tensor[:, :, 1].numpy())
        if legend_labels:
            pos = ax.get_position()
            ax.legend(
                labels=legend_labels, loc="center right", bbox_to_anchor=(1.25, 0.5)
            )
            ax.set_position([pos.x0, pos.y0, pos.width * 0.9, pos.height])
        ax.axis("equal")
        ax.set_title(title)
        if save_path is not None:
            save_plot_and_close(savefig_path=save_path)
        else:
            plt.show()

    def plot_traj(self, traj: Trajectory, save_path: str = None):
        """plot one traj as a 2D top view of x and y positions at all frames

        Args:
            traj (Trajectory): the traj to plot
            save_path (str, optional): where to save the plot. If None we do not save and use plt.show instead. Defaults to None.
        """
        from prescyent.evaluator.plotting import save_plot_and_close

        plt.figure()
        plt.plot(traj.tensor[:, :, 0].numpy(), traj.tensor[:, :, 1].numpy())
        plt.axis("equal")
        plt.title(traj.title)
        if save_path is not None:
            save_plot_and_close(savefig_path=save_path)
        else:
            plt.show()


def generate_noisy_circle(
    radius: float, num_points: int, perturbation_range: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """generate circular coordinates given attributes

    Args:
        radius (float): size of the generated circle
        num_points (int): number of points per circle
        perturbation_range (float): level of noise added

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: angles, x, y
    """
    angles = np.linspace(0, 2 * np.pi, num_points)
    # Adding imperfections to the radius
    perturbations = np.random.uniform(
        -perturbation_range, perturbation_range, num_points
    )
    perturbed_radius = (radius + radius * perturbations).astype("float32")
    # Generating x and y values
    perturbed_circle_x = perturbed_radius * np.cos(angles)
    perturbed_circle_y = perturbed_radius * np.sin(angles)
    # Make last point = first point to have a clean circle
    perturbed_circle_x[-1] = perturbed_circle_x[0]
    perturbed_circle_y[-1] = perturbed_circle_y[0]
    return angles, perturbed_circle_x, perturbed_circle_y
