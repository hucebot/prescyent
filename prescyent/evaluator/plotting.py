"""Util functions for plots"""

from pathlib import Path
from tqdm import tqdm
from typing import Callable, List, Optional, Union

import torch

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.cm import get_cmap


from prescyent.dataset.features.rotation_methods import convert_to_euler
from prescyent.dataset.features.feature import Rotation
from prescyent.dataset import Trajectory
from prescyent.dataset.dataset import MotionDataset
from prescyent.dataset.features.feature_manipulation import cal_distance_for_feat
from prescyent.utils.logger import logger, EVAL


matplotlib.use("agg")


def plot_truth_and_pred(sample, truth, pred, savefig_path=None):
    plt.clf()  # clear just in case
    # we turn shape(seq_len, features) to shape(features, seq_len) to plot the pred by feature
    sample = torch.transpose(sample, 0, 1)
    truth = torch.transpose(truth, 0, 1)
    pred = torch.transpose(pred, 0, 1)
    time_steps = range(len(sample[0]) + len(pred[0]))
    fig, axes = plt.subplots(
        pred.shape[0], sharex=True
    )  # we do one subplot per feature
    for i, axe in enumerate(axes):
        axe.plot(time_steps[: len(sample[i])], sample[i], linewidth=2)
        axe.plot(time_steps[len(sample[i]) :], truth[i], linewidth=2)
        axe.plot(time_steps[len(sample[i]) :], pred[i], linewidth=2, linestyle="--")
    legend_plot(axes, ["Sample", "Truth", "Prediction"])
    fig.set_size_inches(10.5, 10.5)
    fig.suptitle("Motion prediction")
    save_plot_and_close(savefig_path)


def plot_traj_tensors_with_shift(
    trajs,
    savefig_path: str,
    shifts,
    group_labels: List[str] = None,
    traj_labels: List[str] = None,
    dim_labels: List[str] = None,
    title="",
):
    assert len(trajs) > 0
    if group_labels is None:
        group_labels = []
    if traj_labels is None:
        traj_labels = []
    if dim_labels is None:
        dim_labels = []

    # arguments
    if not isinstance(trajs, list):
        trajs = [trajs]
    if not isinstance(shifts, list):
        shifts = [shifts]
    if len(shifts) == 0:
        shifts = [0] * len(trajs)
    if len(traj_labels) == 0:
        traj_labels = [""] * len(trajs)
    if len(group_labels) == 0:
        group_labels = [""] * trajs[0].shape[1]
    if len(dim_labels) == 0:
        dim_labels = [""] * trajs[0].shape[2]
    assert len(traj_labels) == len(trajs)
    assert len(shifts) == len(trajs)
    assert len(group_labels) == trajs[0].shape[1]
    assert len(dim_labels) == trajs[0].shape[2]

    # prepare a subplot for each "group"
    fig, axes = plt.subplots(trajs[0].shape[1], sharex=True)
    if trajs[0].shape[1] == 1:
        axes = [axes]
    fig.set_size_inches(6, trajs[0].shape[1] * 1.5)

    # setup colors
    colors = get_cmap("Accent").colors

    for i, ax in enumerate(axes):  # for each group
        for j, traj in enumerate(trajs):  # for each traj
            ax.set_ylabel(group_labels[i])
            time_steps = range(shifts[j], traj.shape[0] + shifts[j])
            for k in range(traj.shape[2]):
                marker = k % len(Line2D.filled_markers)
                color = colors[j % len(colors)]
                ls = "--" if j != 0 else "-"
                ax.plot(
                    time_steps,
                    traj[:, i, k],
                    linewidth=1,
                    marker=Line2D.filled_markers[marker],
                    markevery=0.1,
                    markersize=2,
                    color=color,
                    ls=ls,
                )

    # tune the look
    for ax in axes:
        ax.minorticks_on()
        ax.grid(color="lightgrey", linestyle="-", lw=0.6)
        ax.grid(which="minor", color="lightgrey", linestyle="--", lw=0.3)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        for spine in ax.spines.values():
            spine.set_position(("outward", 5))
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        ax.tick_params(axis="x", direction="out")
        ax.tick_params(axis="y", length=0)
        ax.tick_params(which="minor", axis="y", length=0)
        ax.spines["bottom"].set_linewidth(2)
        if i != len(axes) - 1:
            ax.spines["bottom"].set_visible(False)
            ax.xaxis.set_ticks_position("none")

    # title
    fig.suptitle(title)

    # legend
    fig.subplots_adjust(right=0.7)
    leg = []
    for j in range(len(trajs)):
        color = colors[j % len(colors)]
        leg += [Line2D([0], [0], color=color, label=traj_labels[j])]
    for k in range(trajs[0].shape[2]):
        marker = k % len(Line2D.filled_markers)
        leg += [
            Line2D(
                [0],
                [0],
                marker=Line2D.filled_markers[marker],
                color="black",
                lw=1,
                label=dim_labels[k],
            )
        ]
    axes[0].legend(handles=leg, bbox_to_anchor=(1.5, 1.1), loc="upper right")

    fig.tight_layout()
    # save the figure
    save_plot_and_close(savefig_path)


def plot_trajectory_prediction(
    trajectory: Trajectory, truth, preds, step: int, savefig_path: str
):
    # we turn shape(seq_len, features) to shape(features, seq_len) to plot the pred by feature
    truth = torch.transpose(truth, 0, 1)
    preds = torch.transpose(preds, 0, 1)

    pred_last_idx = max(len(preds[0]), len(truth[0])) + step

    time_steps = range(pred_last_idx)
    fig, axes = plt.subplots(
        preds.shape[0], sharex=True
    )  # we do one subplot per feature
    if preds.shape[0] == 1:
        axes = [axes]
    for i, axe in enumerate(axes):
        axe.plot(time_steps[: len(truth[i])], truth[i], linewidth=2)
        axe.plot(
            time_steps[: len(preds[i])],
            preds[i],
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
    titles: Optional[List[str]] = None,
):
    assert len(trajectories) >= 1
    feats = trajectories[0].tensor_features
    num_points = trajectories[0].tensor.shape[1]
    num_dims = sum(
        [len(feat.ids) if not isinstance(feat, Rotation) else 3 for feat in feats]
    )
    assert all(
        [traj.tensor_features == feats for traj in trajectories]
    )  # Plotted trajs must have same feats
    assert all(
        [traj.tensor.shape[1] == num_points for traj in trajectories]
    )  # Plotted trajs must have number of points
    pred_last_idx = max(*[len(traj) for traj in trajectories]) + max(*offsets)
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
                if isinstance(feat, Rotation):
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
    fig.suptitle(f"Trajectory and predictions on {trajectories[0].title}")
    # fig.subplots_adjust(right=0.7)
    # fig.tight_layout(pad=5)
    legend_plot(axes, names=titles, xlabel="time (s)", ylabels=ylabels)
    save_plot_and_close(savefig_path)


def plot_multiple_predictors(
    trajectory: Trajectory,
    predictors: List[Callable],
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
                time_steps[pred_last_idx - len(pred[i]) :],
                pred[i],
                linewidth=1,
                linestyle="--",
            )
    legend_plot(
        axes,
        ["Truth"] + [str(predictor) for predictor in predictors],
        ylabels=trajectory.point_names,
    )
    fig.set_size_inches(150, len(trajectory.point_names) + 10)
    fig.suptitle(trajectory.title)
    save_plot_and_close(savefig_path)


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
    legend = axes[-1].legend(
        labels=names, loc="best", bbox_to_anchor=(0.5, 0.0, 0.5, 0.5)
    )
    axes[-1].set_xlabel(xlabel)
    frame = legend.get_frame()
    frame.set_facecolor("0.9")
    frame.set_edgecolor("0.9")
    for i, axe in enumerate(axes):
        if isinstance(ylabels, list) and len(ylabels) >= len(axes):
            axe.set_ylabel(ylabels[i])
        elif ylabels:
            axe.set_ylabel(ylabels[0])
        bottom, top = axe.get_ylim()
        axe.set_ylim(top=round(top, 2) + 0.01, bottom=round(bottom, 2) - 0.01)


def plot_mpjpe(
    predictor: Callable, dataset: MotionDataset, savefig_dir_path: str, log_x=False
):
    distances = list()
    features = dataset.config.out_features
    pbar = tqdm(dataset.test_dataloader)
    pbar.set_description(f"Running {predictor} over test_dataloader:")
    # Run all test once and get distance from truth per feature
    for sample, truth in pbar:
        feat2distances = dict()
        pred = predictor.predict(sample, dataset.config.future_size)
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
            y_values = y_values * 57.2957795
            distance_unit = "degrees"
        plt.xlabel("Time (s)")
        plt.ylabel(f"{feat.name.capitalize()} Mean Error ({distance_unit})")
        plt.grid(True)
        plt.plot(x_values, y_values)
        if log_x:
            plt.gca().set_xscale("log")
        save_plot_and_close(f"{savefig_dir_path}MPJE_{feat.name}.pdf")


def plot_mpjpes(
    predictors: List[Callable],
    dataset: MotionDataset,
    savefig_dir_path: str,
    log_x=False,
):
    predictors_distances = list()
    features = dataset.config.out_features
    for predictor in predictors:
        distances = list()
        pbar = tqdm(dataset.test_dataloader)
        pbar.set_description(f"Running {predictor} over test_dataloader:")
        # Run all test once and get distance from truth per feature
        for sample, truth in pbar:
            feat2distances = dict()
            pred = predictor.predict(sample, dataset.config.future_size)
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
        for distances in predictors_distances:
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
                y_values = y_values * 57.2957795
                distance_unit = "degrees"
            plt.plot(x_values, y_values)

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
