"""Util functions for plots"""

from pathlib import Path
from typing import Iterator, Tuple

import matplotlib.pyplot as plt
import torch

from prescyent.dataset.motion.datasamples import MotionDataSamples


def plot_data(data: Iterator, savefig_path=None):
    # if not isinstance(data, np.ndarray):
    #     data = np.fromiter(data)
    plt.clf()
    plt.plot(data)
    plt.legend()
    if savefig_path is not None:
        save_fig_util(savefig_path)


def plot_datasample(data_sample: Tuple[torch.Tensor, torch.Tensor], savefig_path=None):
    sample, truth = data_sample
    plt.clf()
    plt.plot(torch.cat((sample, truth)))
    plt.legend()
    if savefig_path is not None:
        save_fig_util(savefig_path)


def plot_prediction(data_sample: Tuple[torch.Tensor, torch.Tensor],
                    pred: torch.Tensor, savefig_path=None):
    plt.clf()   # clear just in case
    sample, truth = data_sample
    # we turn shape(seq_len, features) to shape(features, seq_len) to plot the pred by feature
    sample = torch.swapaxes(sample, 0, 1)
    truth = torch.swapaxes(truth, 0, 1)
    pred = torch.swapaxes(pred, 0, 1)
    x = range(len(sample[0]) + len(truth[0]))
    fig, axes = plt.subplots(pred.shape[0], sharex=True)  # we do one subplot per feature
    fig.suptitle('Motion Prediction plots')
    for i, axe in enumerate(axes):
        axe.plot(x, torch.cat((sample[i], truth[i])), linewidth=2)
        axe.plot(x[len(sample[i]):], pred[i], linewidth=2, linestyle='--')
    # plt.plot(x, torch.cat((sample, truth)), linewidth=2, color='#B22400')
    # plt.plot(x[len(sample):], pred, linewidth=2, linestyle='--', color='#006BB2')
    legend = plt.legend(["Truth", "Prediction"], loc=3)
    frame = legend.get_frame()
    frame.set_facecolor('0.9')
    frame.set_edgecolor('0.9')
    if savefig_path is not None:
        save_fig_util(savefig_path)


def save_fig_util(savefig_path):
    if not Path(savefig_path).parent.exists():
        Path(savefig_path).parent.mkdir(parents=True)
    plt.savefig(savefig_path)
