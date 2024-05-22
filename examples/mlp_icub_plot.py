import glob
import os
import sys

import torch
from tqdm import tqdm

from prescyent.predictor import MlpPredictor
from prescyent.dataset import TeleopIcubDataset
from prescyent.evaluator.plotting import plot_traj_tensors_with_shift


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] == "--help":
        print("Usage (examples):")
        print("plot_mlp.py --model data/models/teleopicub/all/MlpPredictor/version_0")
        print("plot_mlp.py --last data/models/teleopicub/all/MlpPredictor/")
        sys.exit(1)

    if sys.argv[1] == "--model":
        path = sys.argv[2]
    elif sys.argv[1] == "--last":
        path = max(glob.glob(sys.argv[2] + "/version_*"), key=os.path.getctime)
    else:
        print("Error: use --last or --model")
        sys.exit(1)
    print("Path:", path)
    # we load with the same config as for the training
    print("Loading the dataset...")
    dataset = TeleopIcubDataset(path + "/dataset.config", load_data_at_init=True)
    history_size = dataset.config.history_size
    future_size = dataset.config.future_size
    print("Dataset OK")

    # load a pretrained model
    print("Loading the predictor...")
    predictor = MlpPredictor(path)
    print("Predictor OK")

    # predict the trajectories
    print(
        f"Computing predictions for all the test trajectories for {predictor.name}..."
    )
    all_preds = []

    # for traj in tqdm(dataset.trajectories.test):  # for each test trajectories
    #     prediction = run_predictor(predictor, traj.tensor, history_size, future_size, "windowed")
    #     all_preds.append(prediction)
    with torch.no_grad():
        for traj in tqdm(dataset.trajectories.test):  # for each test trajectories
            pred = torch.zeros(
                (traj.shape[0] - history_size, traj.shape[1], traj.shape[2])
            )
            for i in range(0, traj.shape[0] - history_size):  # for each time-step
                # --- this is the prediction
                p = predictor.predict(traj[i : i + history_size, :, :], future_size)
                # ---
                # We keep last frame of the predicted sequence (so {history_size + future_size} seconds from first input)
                pred[i] = p[-1]
            all_preds += [pred]
    print("Predictions OK")

    # plot everything in the current directory
    print("Plotting...")
    for i in tqdm(range(len(all_preds))):  # for each test trtrzajectory
        ref_traj = dataset.trajectories.test[i]
        title = "/".join(ref_traj.file_path.parts[-2:])
        dims = ref_traj.point_names
        plot_traj_tensors_with_shift(
            [ref_traj.tensor, all_preds[i], ref_traj.tensor],
            f"examples/data/plots/pred_{i}.pdf",
            # we shift the predictions of the history_size (data we need to predict)
            # and of future_size (because given 1 s of data, we predict 1 second from now)
            shifts=[0, history_size + future_size - 1, future_size],
            group_labels=dims,
            dim_labels=ref_traj.dim_names,
            title=title,
            traj_labels=["Truth", "Prediction", "delayed (1s)"],
        )
