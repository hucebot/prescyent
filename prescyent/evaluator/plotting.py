"""Util functions for plots"""

from pathlib import Path
from typing import Callable, List, Tuple, Union

import matplotlib.pyplot as plt
import torch
from matplotlib.axes import Axes

from prescyent.dataset.motion.episodes import Episode


def plot_prediction(data_sample: Tuple[torch.Tensor, torch.Tensor],
                    pred: torch.Tensor, savefig_path=None):
    """plot input, truth and pred

    Args:
        data_sample (Tuple[torch.Tensor, torch.Tensor]):  tuple(input, truth)
        pred (torch.Tensor): prediction
        savefig_path (_type_, optional): if there is a path we save. Defaults to None.
    """
    plt.clf()   # clear just in case
    sample, truth = data_sample
    # we turn shape(seq_len, features) to shape(features, seq_len) to plot the pred by feature
    sample = torch.transpose(sample, 0, 1)
    truth = torch.transpose(truth, 0, 1)
    pred = torch.transpose(pred, 0, 1)
    timesteps = range(len(sample[0]) + len(truth[0]))
    fig, axes = plt.subplots(pred.shape[0], sharex=True)  # we do one subplot per feature
    for i, axe in enumerate(axes):
        axe.plot(timesteps, torch.cat((sample[i], truth[i])), linewidth=2)
        axe.plot(timesteps[len(sample[i]):], pred[i], linewidth=2, linestyle='--')
    legend_plot(axes, ["Truth", "Prediction"])
    fig.set_size_inches(10.5, 10.5)
    fig.suptitle("Motion prediction")
    save_plot_and_close(savefig_path)


def plot_episode_prediction(episode, inputs, preds, step, savefig_path, eval_on_last_pred):
    # we turn shape(seq_len, features) to shape(features, seq_len) to plot the pred by feature
    inputs = torch.transpose(inputs, 0, 1)
    preds = torch.transpose(preds, 0, 1)

    pred_last_idx = len(inputs[0]) + step

    timesteps = range(pred_last_idx)
    fig, axes = plt.subplots(preds.shape[0], sharex=True)  # we do one subplot per feature
    for i, axe in enumerate(axes):
        axe.plot(timesteps[:-step], inputs[i], linewidth=2)
        if eval_on_last_pred:
            # fancy lines to range from last of first prediction (2 step - 1)
            # to the last predicted index (len + step -1)
            stepped_x = list(range(2 * step - 1, pred_last_idx, step))
            if pred_last_idx - 1 not in stepped_x:
                stepped_x.append(pred_last_idx - 1)
            axe.plot(stepped_x, preds[i], linewidth=2, linestyle='--')
        else:
            axe.plot(timesteps[step:], preds[i], linewidth=2, linestyle='--')
        axe.plot(timesteps[step:], inputs[i], linewidth=2)
    legend_plot(axes, ["Truth", "Prediction", "Delayed Truth"],
                ylabels=episode.dimension_names)
    fig.set_size_inches(pred_last_idx / 20, len(episode.dimension_names))
    fig.suptitle(episode.file_path)
    save_plot_and_close(savefig_path)


def plot_multiple_predictors(episode: Episode,
                             predictors: List[Callable],
                             predictions: List[torch.Tensor],
                             step: int, savefig_path: str):
    # we turn shape(seq_len, features) to shape(features, seq_len) to plot the pred by feature
    truth = torch.transpose(episode.tensor, 0, 1)
    preds = [torch.transpose(pred, 0, 1) for pred in predictions]
    pred_last_idx = len(episode) + step
    timesteps = range(pred_last_idx)
    # we do one subplot per feature
    fig, axes = plt.subplots(truth.shape[0], sharex=True)
    for i, axe in enumerate(axes):
        axe.plot(timesteps[:-step], truth[i], linewidth=2)
        for pred in preds:
            axe.plot(timesteps[pred_last_idx - len(pred[i]):], pred[i], linewidth=1, linestyle='--')
    legend_plot(axes, ["Truth"] + [str(predictor) for predictor in predictors],
                ylabels=episode.dimension_names)
    fig.set_size_inches(pred_last_idx / 15, len(episode.dimension_names) + 2)
    fig.suptitle(episode.file_path)
    save_plot_and_close(savefig_path)


def save_plot_and_close(savefig_path):
    """savefig helper"""
    if savefig_path is not None:
        if not Path(savefig_path).parent.exists():
            Path(savefig_path).parent.mkdir(parents=True)
        plt.savefig(savefig_path)
    plt.close()


def legend_plot(axes: List[Axes],
                names: List[str],
                xlabel: str = "time",
                ylabels: Union[List[str], str] = "pos"):
    """standardized lengend function for all plots of the library

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
