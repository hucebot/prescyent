"""Util functions for plots"""

from pathlib import Path
from typing import Callable, List, Union

import matplotlib
import matplotlib.pyplot as plt
import torch
from matplotlib.axes import Axes

from prescyent.dataset.trajectories import Trajectory
from prescyent.utils.logger import logger, EVAL


matplotlib.use('agg')


def plot_truth_and_pred(sample, truth, pred, savefig_path=None):
    plt.clf()   # clear just in case
    # we turn shape(seq_len, features) to shape(features, seq_len) to plot the pred by feature
    sample = torch.transpose(sample, 0, 1)
    truth = torch.transpose(truth, 0, 1)
    pred = torch.transpose(pred, 0, 1)
    time_steps = range(len(sample[0]) + len(pred[0]))
    fig, axes = plt.subplots(pred.shape[0], sharex=True)  # we do one subplot per feature
    for i, axe in enumerate(axes):
        axe.plot(time_steps[:len(sample[i])], sample[i], linewidth=2)
        axe.plot(time_steps[len(sample[i]):], truth[i], linewidth=2)
        axe.plot(time_steps[len(sample[i]):], pred[i], linewidth=2, linestyle='--')
    legend_plot(axes, ["Sample", "Truth", "Prediction"])
    fig.set_size_inches(10.5, 10.5)
    fig.suptitle("Motion prediction")
    save_plot_and_close(savefig_path)


def plot_trajectory_prediction(trajectory, preds, step, savefig_path):
    # we turn shape(seq_len, features) to shape(features, seq_len) to plot the pred by feature
    inputs = torch.transpose(trajectory.tensor, 0, 1)
    preds = torch.transpose(preds, 0, 1)

    pred_last_idx = max(len(preds[0]), len(inputs[0])) + step

    time_steps = range(pred_last_idx)
    fig, axes = plt.subplots(preds.shape[0], sharex=True)  # we do one subplot per feature
    if preds.shape[0] == 1:
        axes = [axes]
    for i, axe in enumerate(axes):
        axe.plot(time_steps[:len(inputs[i])], inputs[i], linewidth=2)
        axe.plot(time_steps[-len(inputs[i]):], inputs[i][:min(len(inputs[i]), pred_last_idx)], linewidth=2)  # delayed
        axe.plot(time_steps[step:step + len(preds[i])], preds[i], linewidth=2, linestyle='--')
    legend_plot(axes, ["Truth", "Delayed Truth", "Prediction"],
                ylabels=trajectory.dimension_names)
    fig.set_size_inches(15, len(trajectory.dimension_names))
    fig.suptitle(trajectory.file_path)
    save_plot_and_close(savefig_path)


def plot_multiple_predictors(trajectory: Trajectory,
                             predictors: List[Callable],
                             predictions: List[torch.Tensor],
                             step: int, savefig_path: str):
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
        axe.plot(time_steps[:len(truth[i])], truth[i], linewidth=2)
        for pred in preds:
            axe.plot(time_steps[pred_last_idx - len(pred[i]):],
                     pred[i], linewidth=1, linestyle='--')
    legend_plot(axes, ["Truth"] + [str(predictor) for predictor in predictors],
                ylabels=trajectory.dimension_names)
    fig.set_size_inches(15, len(trajectory.dimension_names) + 2)
    fig.suptitle(trajectory.file_path)
    save_plot_and_close(savefig_path)

def plot_multiple_future(future_sizes: List[int],
                         trajectory: Trajectory,
                         predictor: Callable,
                         predictions: List[torch.Tensor],
                         step: int, savefig_path: str):
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
        axe.plot(time_steps[:len(truth[i])], truth[i], linewidth=2)
        for pred in preds:
            axe.plot(time_steps[step:step+len(pred[i])],
                     pred[i], linewidth=1, linestyle='--')
    legend_plot(axes, ["Truth"] + [str(predictor) + f"f_{future}" for future in future_sizes],
                ylabels=trajectory.dimension_names)
    fig.set_size_inches(15, len(trajectory.dimension_names) + 2)
    fig.suptitle(trajectory.file_path)
    save_plot_and_close(savefig_path)


def save_plot_and_close(savefig_path):
    """savefig helper"""
    if savefig_path is not None:
        if not Path(savefig_path).parent.exists():
            Path(savefig_path).parent.mkdir(parents=True)
        plt.savefig(savefig_path)
        logger.info("Saving plot to %s", savefig_path, group=EVAL)
    plt.close()


def legend_plot(axes: List[Axes],
                names: List[str],
                xlabel: str = "time",
                ylabels: Union[List[str], str] = "pos"):
    """standardized legend function for all plots of the library

    Args:
        axes (List[Axes]): axes to describe
        names (List[str]): legend names of the plot
        xlabel (str, optional): label for x. x axis are shared in our plots. Defaults to "time".
        ylabels (List[str], optional): labels for y. Defaults to ["pos"].
    """
    legend = axes[-1].legend(names, loc=1)
    axes[-1].set_xlabel(xlabel)
    frame = legend.get_frame()
    frame.set_facecolor('0.9')
    frame.set_edgecolor('0.9')
    for i, axe in enumerate(axes):
        if isinstance(ylabels, list) and len(ylabels) >= len(axes):
            axe.set_ylabel(ylabels[i])
        elif ylabels:
            axe.set_ylabel(ylabels[0])
        bottom, top = axe.get_ylim()
        axe.set_ylim(top=round(top, 2) + .01,
                     bottom=round(bottom, 2) - .01)
