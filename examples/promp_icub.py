# this example shows how to learn a set of ProMPs (Probabilistic Motion Primitives)
#  on the TeleopIcubDataset
from tqdm import tqdm
import torch
import numpy as np

from prescyent.predictor.promp import PrompPredictor, PrompConfig
from prescyent.dataset import TeleopIcubDataset, TeleopIcubDatasetConfig
from prescyent.dataset.features import CoordinateXYZ

import matplotlib.pyplot as plt


if __name__ == "__main__":
    print("Initializing dataset...", end=" ")
    frequency: int = 10  # subsampling -> 100 Hz to 10Hz
    history_size = 10  # 1 second
    future_size = 10  # 1 second
    # for TeleopIcub dimension = [0, 1, 2] is [waist, right_hand, left_hand]
    features = CoordinateXYZ(range(3))
    batch_size = 256
    dataset_config = TeleopIcubDatasetConfig(
        history_size=history_size,  # not used
        future_size=future_size,  # not used
        frequency=frequency,
        batch_size=batch_size,  # not used
        in_features=features,
        out_features=features,
        subsets=["datasetMultipleTasks/BottleTable"],  # modified from default
    )
    print("OK")

    print("Loading dataset...", end="")
    dataset = TeleopIcubDataset(dataset_config)
    print("OK")

    print("Initializing predictor...", end="")
    predictor = PrompPredictor(PrompConfig(dataset_config=dataset_config))
    print("OK")

    print("Training...", end="")
    predictor.train(dataset)
    print("OK")

    print("Conditioning")
    predictor.promps[1][1].plot("cond_")
    traj = dataset.trajectories.test[0].tensor[:, 1, 1]
    p = predictor.promps[1][1].condition(traj[0:50])
    p.plot("cond_")

    print("Testing...", end="")
    for p in range(dataset.trajectories.test[0].tensor.size(1)):
        for n in range(dataset.trajectories.test[0].tensor.size(2)):
            fig, ax = plt.subplots()
            for i, traj in enumerate(dataset.trajectories.test):
                ax.plot(
                    np.arange(0, traj.tensor.size(0)),
                    traj.tensor[:, p, n],
                    label=str(i),
                )
            fig.legend()
            fig.savefig("test_" + str(p) + "_" + str(n) + ".pdf")

    result_list = []
    std_list = []
    for traj in tqdm(dataset.trajectories.test):
        result = torch.zeros_like(traj.tensor)
        std = torch.zeros_like(traj.tensor)
        for t in tqdm(range(0, traj.tensor.size(0)), colour="blue"):
            p, sigma = predictor.predict_by_conditioning(
                traj.tensor[0:t, :, :], future_size
            )
            result[t, :, :] = p
            std[t, :, :] = sigma
        result_list += [result]
        std_list += [std]
    print("OK")

    print("Plotting...", end="")
    for i, (pred, std, gt) in tqdm(
        enumerate(zip(result_list, std_list, dataset.trajectories.test))
    ):
        for p in range(pred.size(1)):
            for n in range(pred.size(2)):
                fig, ax = plt.subplots()
                x_pred = np.arange(0, pred.size(0)) + future_size
                ax.fill_between(
                    x_pred,
                    pred[:, p, n] - std[:, p, n],
                    pred[:, p, n] + std[:, p, n],
                    color="red",
                    alpha=0.25,
                )
                ax.plot(x_pred, pred[:, p, n], label="prediction", color="red")
                ax.plot(
                    np.arange(0, gt.tensor.size(0)),
                    gt.tensor[:, p, n],
                    label="ground truth",
                    color="green",
                )
                mean_promp = predictor.promps[p][n].mean()
                std_promp = predictor.promps[p][n].std()
                x = np.arange(0, mean_promp.size(0))
                ax.fill_between(
                    x,
                    mean_promp - std_promp,
                    mean_promp + std_promp,
                    color="grey",
                    alpha=0.25,
                )
                ax.plot(x, mean_promp, label="mean", color="grey", lw=3)
                ax.legend()
                fig.savefig(f"pred_{i}_{p}_{n}.pdf")
    print("OK")
