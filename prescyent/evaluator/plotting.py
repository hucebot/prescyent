"""Util functions for plots"""

from math import pi as math_pi
from pathlib import Path
from typing import Callable, List, Optional, Union

import numpy as np
import torch
from tqdm import tqdm
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from prescyent.dataset.features.rotation_methods import convert_to_euler
from prescyent.dataset.features.feature import Rotation
from prescyent.dataset import Trajectory
from prescyent.dataset.dataset import TrajectoriesDataset
from prescyent.dataset.features.feature_manipulation import cal_distance_for_feat
from prescyent.predictor.base_predictor import BasePredictor
from prescyent.utils.enums import LearningTypes
from prescyent.utils.logger import logger, EVAL


def plot_trajectory_prediction(
    trajectory: Trajectory, truth, pred, overprediction: int, savefig_path: str
):
    """Kinda naive plotting that can become messy if there is a lot of features or points
    Truth and pred's frame must already be aligned,
    and the amount of frames predicted that aren't in truth must appear in overprediction
    """

    # we turn shape(seq_len, features) to shape(features, seq_len) to plot the pred by feature
    truth = torch.transpose(truth, 0, 1)
    pred = torch.transpose(pred, 0, 1)

    pred_last_idx = max(len(pred[0]), len(truth[0])) + overprediction

    time_steps = range(pred_last_idx)
    fig, axes = plt.subplots(
        pred.shape[0], sharex=True
    )  # we do one subplot per feature
    if pred.shape[0] == 1:
        axes = [axes]
    for i, axe in enumerate(axes):
        axe.plot(time_steps[: len(truth[i])], truth[i], linewidth=2)
        axe.plot(
            time_steps[: len(pred[i])],
            pred[i],
            linewidth=1,
            linestyle="--",
        )
        axe.grid(color="grey", linestyle="--", lw=0.3)
        axe.set_axisbelow(True)
        axe.spines["top"].set_visible(False)
        axe.spines["right"].set_visible(False)
        axe.spines["left"].set_visible(False)
        for spine in axe.spines.values():
            spine.set_position(("outward", 5))
        axe.get_xaxis().tick_bottom()
        axe.get_yaxis().tick_left()
        axe.tick_params(axis="x", direction="out")
        axe.tick_params(axis="y", length=0)
        axe.tick_params(which="minor", axis="y", length=0)
        axe.spines["bottom"].set_linewidth(2)
        if i != len(axes) - 1:
            axe.spines["bottom"].set_visible(False)
            axe.xaxis.set_ticks_position("none")
    legend_plot(
        axes,
        trajectory.dim_names,
        ylabels=trajectory.point_names,
    )
    fig.set_size_inches(
        trajectory.tensor.shape[1] * 2 + 5, len(trajectory.point_names) + 5
    )
    fig.suptitle(trajectory.title)
    fig.subplots_adjust(right=0.7)
    fig.tight_layout()
    save_plot_and_close(savefig_path)


def plot_trajs(
    trajectories: List[Trajectory],
    offsets: List[int],
    savefig_path: str,
    title: Optional[str] = None,
    legend_labels: Optional[List[str]] = None,
    rot_to_euler: bool = True,
):
    """plot a list of trajectories

    Args:
        trajectories (List[Trajectory]): list of N trajectories
        offsets (List[int]): list of N offsets to shift trajectories[N] over offsets[N] frames
        savefig_path (str): path where to save the plot
        title (Optional[str], optional): title of the plot. If none, the title is the title of trajectories[0]. Defaults to None.
        legend_labels (Optional[List[str]], optional): labels that will serve as the legend of the plot. If None, No legend. Defaults to None.
        rot_to_euler (bool, optional): if true, convert any rotation to euler for plotting with 3D. Defaults to True.
    """

    assert len(trajectories) >= 1
    feats = trajectories[0].tensor_features
    num_points = trajectories[0].tensor.shape[1]
    if rot_to_euler:
        num_dims = sum(
            [len(feat.ids) if not isinstance(feat, Rotation) else 3 for feat in feats]
        )
    else:
        num_dims = sum([len(feat.ids) for feat in feats])
    assert all(
        [traj.tensor_features == feats for traj in trajectories]
    )  # Plotted trajs must have same feats
    assert all(
        [traj.tensor.shape[1] == num_points for traj in trajectories]
    )  # Plotted trajs must have number of points
    pred_last_idx = max([len(traj) for traj in trajectories]) + max(offsets)
    time_steps = np.linspace(
        0,
        (pred_last_idx + 1) / trajectories[0].frequency,
        pred_last_idx,
        endpoint=False,
    )
    fig, axes = plt.subplots(
        num_dims * num_points, sharex=True
    )  # we do one subplot per dim and per point
    if num_dims * num_points == 1:
        axes = [axes]
    axe_id = 0
    ylabels = []
    for point in range(num_points):
        for feat in feats:
            for offset, traj in zip(offsets, trajectories):
                if isinstance(feat, Rotation) and rot_to_euler:
                    feat_tensor = convert_to_euler(traj.tensor[:, point, feat.ids])
                    dims_names = ["roll", "pitch", "yaw"]
                else:
                    feat_tensor = traj.tensor[:, point, feat.ids]
                    dims_names = feat.dims_names
                # (seq_len, num_dims) => (num_dims, seq_len)
                feat_tensor = torch.transpose(feat_tensor, 0, 1)
                for dim_id, dim_tensor in enumerate(feat_tensor):
                    # change axe for each dim
                    axes[axe_id + dim_id].plot(
                        time_steps[offset : len(dim_tensor) + offset],
                        dim_tensor,
                        linewidth=0.5,
                    )
            ylabels += [
                f"{trajectories[0].point_names[point]}_{dim_name}"
                for dim_name in dims_names
            ]
            axe_id += len(feat_tensor)  # add dim size to used axes
    w = min(
        pred_last_idx * 0.025 + 5, 2**16 / 100 - 1
    )  # caculated values or max value accepted by matplotlib (max is 2¹⁶ pxl and default dpi is 100)
    h = min(
        len(axes) * 5 + 5, 2**16 / 100 - 1
    )  # caculated values or max value accepted by matplotlib (max is 2¹⁶ pxl and default dpi is 100)
    fig.set_size_inches(w, h)
    if title is None:
        title = f"Trajectory and predictions on {trajectories[0].title}"
    fig.suptitle(title)
    # fig.subplots_adjust(right=0.7)
    fig.tight_layout(pad=10)
    legend_plot(axes, names=legend_labels, xlabel="time (s)", ylabels=ylabels)
    save_plot_and_close(savefig_path)


def plot_multiple_predictors(
    trajectory: Trajectory,
    predictors: List[BasePredictor],
    savefig_path: str,
):
    """given a trajectory and a list of predictors, plots the predictions and truth

    Args:
        trajectory (Trajectory): trajectory to predict over
        predictors (List[BasePredictor]): list of predictors which prediction's we will compare
        savefig_path (str): path where to save the plot

    Raises:
        AttributeError: All predictors must share the same dataset config to be compared
    """

    if not len(set([p.config.dataset_config for p in predictors])):
        raise AttributeError(
            "All predictors must share the same dataset config to be compared"
        )
    dataset_config = predictors[0].config.dataset_config
    input_traj = trajectory.create_subtraj(
        dataset_config.in_points,
        dataset_config.in_features,
        dataset_config.context_keys,
    )
    truth = trajectory.create_subtraj(
        dataset_config.out_points, dataset_config.out_features
    )
    trajs, offsets = [], []
    for predictor in predictors:
        traj, offset = predictor.predict_trajectory(input_traj)
        trajs.append(traj)
        offsets.append(offset)
    plot_trajs(
        trajectories=[truth] + trajs,
        offsets=[0] + offsets,
        savefig_path=savefig_path,
        title=trajectory.title,
        legend_labels=["Truth"] + [str(p) for p in predictors],
        rot_to_euler=True,
    )


def plot_multiple_future(
    future_sizes: List[int],
    trajectory: Trajectory,
    predictor: Callable,
    predictions: List[torch.Tensor],
    step: int,
    savefig_path: str,
):
    pred_last_idx = max([len(pred) for pred in predictions]) + step
    # we turn shape(seq_len, features) to shape(features, seq_len) to plot the pred by feature
    truth = torch.transpose(trajectory.tensor, 0, 1)
    preds = [torch.transpose(pred, 0, 1) for pred in predictions]
    time_steps = range(pred_last_idx)
    # we do one subplot per feature
    fig, axes = plt.subplots(truth.shape[0], sharex=True)
    if preds[0].shape[0] == 1:
        axes = [axes]
    for i, axe in enumerate(axes):
        axe.plot(time_steps[: len(truth[i])], truth[i], linewidth=2)
        for pred in preds:
            axe.plot(
                time_steps[step : step + len(pred[i])],
                pred[i],
                linewidth=1,
                linestyle="--",
            )
    legend_plot(
        axes,
        ["Truth"] + [str(predictor) + f"f_{future}" for future in future_sizes],
        ylabels=trajectory.point_names,
    )
    fig.set_size_inches(15, len(trajectory.point_names) + 2)
    fig.suptitle(trajectory.title)
    save_plot_and_close(savefig_path)


def save_plot_and_close(savefig_path, dpi=100):
    """savefig helper"""
    if savefig_path is not None:
        if not Path(savefig_path).parent.exists():
            Path(savefig_path).parent.mkdir(parents=True)
        plt.savefig(savefig_path, dpi=dpi)
        logger.getChild(EVAL).info("Saving plot to %s", savefig_path)
    plt.close()


def legend_plot(
    axes: List[Axes],
    names: List[str],
    xlabel: str = "time",
    ylabels: Union[List[str], str] = "pos",
):
    """standardized legend function for all plots of the library

    Args:
        axes (List[Axes]): axes to describe
        names (List[str]): legend names of the plot
        xlabel (str, optional): label for x. x axis are shared in our plots. Defaults to "time".
        ylabels (List[str], optional): labels for y. Defaults to ["pos"].
    """
    if names is not None:
        legend = axes[-1].legend(labels=names, loc="best")
        frame = legend.get_frame()
        frame.set_facecolor("0.9")
        frame.set_edgecolor("0.9")
    axes[-1].set_xlabel(xlabel)
    for i, axe in enumerate(axes):
        if isinstance(ylabels, list) and len(ylabels) >= len(axes):
            axe.set_ylabel(ylabels[i])
        elif ylabels:
            axe.set_ylabel(ylabels[0])
        bottom, top = axe.get_ylim()
        axe.set_ylim(top=round(top, 2) + 0.01, bottom=round(bottom, 2) - 0.01)


def plot_mpjpe(
    predictor: Callable,
    dataset: TrajectoriesDataset,
    savefig_dir_path: str,
    log_x=False,
):
    """Plot the MPJPE evaluation of the predictor

    Args:
        predictor (Callable): predictor to test
        dataset (TrajectoriesDataset): dataset instance
        savefig_dir_path (str): path where to save the plot
        log_x (bool, optional): if true, use log scale. Defaults to False.
    """
    if predictor.config.dataset_config.learning_type == LearningTypes.SEQ2ONE:
        logger.getChild(EVAL).warning(
            f"Cannot compute per frame evaluation of SEQ2ONE predictor {predictor}. Simlpy refer to ADE or FDE."
        )
        return
    distances = list()
    features = dataset.config.out_features
    # Run all test once and get distance from truth per feature
    if dataset.config.learning_type == LearningTypes.AUTOREG:
        dataset.config.learning_type = LearningTypes.SEQ2SEQ
        dataset.generate_samples("test")
    pbar = tqdm(dataset.test_dataloader(), colour="green")
    pbar.set_description(f"Running {predictor} over test_dataloader:")
    for sample, context, truth in pbar:
        feat2distances = dict()
        pred = predictor.predict(sample, dataset.config.future_size, context)
        for feat in features:
            feat2distances[feat.name] = cal_distance_for_feat(
                pred[..., feat.ids], truth[..., feat.ids], feat
            ).detach()
        distances.append(feat2distances)
    # Plot mean MPJPE per feature
    for feat in features:
        batch_feat_distances = torch.cat(
            [feat2distances[feat.name] for feat2distances in distances]
        )
        mpjpe = (
            batch_feat_distances.transpose(0, 1)
            .reshape(dataset.config.future_size, -1)
            .mean(-1)
        )
        y_values = mpjpe.numpy()
        x_max = dataset.config.future_size / dataset.frequency
        x_values = np.flip(np.linspace(x_max, 0, len(y_values), endpoint=False))
        distance_unit = feat.distance_unit
        if distance_unit == "rad":
            y_values = y_values * 180 / math_pi
            distance_unit = "degrees"
        plt.xlabel("Time (s)")
        plt.ylabel(f"{feat.name.capitalize()} Mean Error ({distance_unit})")
        plt.grid(True)
        plt.plot(x_values, y_values)
        if log_x:
            plt.gca().set_xscale("log")
        logger.getChild(EVAL).info(f"MPJPE: {y_values}")
        save_plot_and_close(f"{savefig_dir_path}/MPJE_{feat.name}.pdf")


def plot_mpjpes(
    predictors: List[Callable],
    dataset: TrajectoriesDataset,
    savefig_dir_path: str,
    log_x=False,
):
    """Plot the MPJPE evaluation of the predictor

    Args:
        predictor (List[Callable]): list of predictors to test
        dataset (TrajectoriesDataset): dataset instance
        savefig_dir_path (str): path where to save the plot
        log_x (bool, optional): if true, use log scale. Defaults to False.
    """
    predictors_distances = list()
    features = dataset.config.out_features
    for predictor in predictors:
        if predictor.config.dataset_config.learning_type == LearningTypes.SEQ2ONE:
            logger.getChild(EVAL).warning(
                f"Cannot compute per frame evaluation of SEQ2ONE predictor {predictor}. Simlpy refer to ADE or FDE."
            )
            continue
        distances = list()
        pbar = tqdm(dataset.test_dataloader(), colour="green")
        pbar.set_description(f"Running {predictor} over test_dataloader:")
        # Run all test once and get distance from truth per feature
        for sample, context, truth in pbar:
            feat2distances = dict()
            pred = predictor.predict(sample, dataset.config.future_size, context)
            for feat in features:
                feat2distances[feat.name] = cal_distance_for_feat(
                    pred[..., feat.ids], truth[..., feat.ids], feat
                ).detach()
            distances.append(feat2distances)
        predictors_distances.append(distances)
    # Plot mean MPJPE per feature
    for feat in features:
        x_max = dataset.config.future_size / dataset.frequency
        x_values = np.flip(
            np.linspace(x_max, 0, dataset.config.future_size, endpoint=False)
        )
        for d, distances in enumerate(predictors_distances):
            batch_feat_distances = torch.cat(
                [feat2distances[feat.name] for feat2distances in distances]
            )
            mpjpe = (
                batch_feat_distances.transpose(0, 1)
                .reshape(dataset.config.future_size, -1)
                .mean(-1)
            )
            y_values = mpjpe.numpy()
            distance_unit = feat.distance_unit
            if distance_unit == "rad":
                y_values = y_values * 180 / math_pi
                distance_unit = "degrees"
            plt.plot(x_values, y_values)
            logger.getChild(EVAL).info(f"MPJPE for {predictors[d]}: {y_values}")

        plt.xlabel("Time (s)")
        plt.ylabel(f"{feat.name.capitalize()} Mean Error ({distance_unit})")
        plt.grid(True)
        legend = plt.legend([p.name for p in predictors], loc=4)
        frame = legend.get_frame()
        frame.set_facecolor("0.9")
        frame.set_edgecolor("0.9")
        if log_x:
            plt.gca().set_xscale("log")
        save_plot_and_close(f"{savefig_dir_path}/MPJE_{feat.name}.pdf")
