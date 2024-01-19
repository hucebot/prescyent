"""Util functions for plots"""

from pathlib import Path
from typing import Callable, List, Union

import torch

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.cm import get_cmap

from prescyent.dataset import Trajectory
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


def plot_trajs(
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
    trajectory: Trajectory, preds, step: int, savefig_path: str
):
    # we turn shape(seq_len, features) to shape(features, seq_len) to plot the pred by feature
    inputs = torch.transpose(trajectory.tensor, 0, 1)
    preds = torch.transpose(preds, 0, 1)

    pred_last_idx = max(len(preds[0]), len(inputs[0])) + step

    time_steps = range(pred_last_idx)
    fig, axes = plt.subplots(
        preds.shape[0], sharex=True
    )  # we do one subplot per feature
    if preds.shape[0] == 1:
        axes = [axes]
    for i, axe in enumerate(axes):
        axe.plot(time_steps[: len(inputs[i])], inputs[i], linewidth=2)
        # axe.plot(time_steps[-len(inputs[i]):],
        # inputs[i][:min(len(inputs[i]),
        # pred_last_idx)], linewidth=2)  # delayed
        axe.plot(
            time_steps[step : step + len(preds[i])],
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
        [
            "Truth_x",
            "Truth_y",
            "Truth_z",
            "Prediction_x",
            "Prediction_y",
            "Prediction_z",
        ],
        ylabels=trajectory.point_names,
    )
    fig.set_size_inches(
        trajectory.tensor.shape[1] * 2 + 5, len(trajectory.point_names) + 5
    )
    fig.suptitle(trajectory.title)
    fig.subplots_adjust(right=0.7)
    fig.tight_layout()
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


def save_plot_and_close(savefig_path, dpi=300):
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
    legend = axes[-1].legend(names, bbox_to_anchor=(1.45, 1.1), loc="lower right")
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
