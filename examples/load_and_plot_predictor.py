from argparse import ArgumentParser
from pathlib import Path

from tqdm import tqdm

from prescyent.auto_predictor import AutoPredictor
from prescyent.auto_dataset import AutoDataset
from prescyent.evaluator.plotting import plot_trajs


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("model_path", help="data/models/teleopicub/all/MlpPredictor/version_0")
    args = parser.parse_args()

    path = Path(args.model_path)
    print("Path:", path)
    # we load with the same config as for the training
    print("Loading the dataset...")
    dataset = AutoDataset.build_from_config(path)
    history_size = dataset.config.history_size
    future_size = dataset.config.future_size
    print("Dataset OK")

    # load a pretrained model
    print("Loading the predictor...")
    predictor = AutoPredictor.load_from_config(path)
    print("Predictor OK")

    # predict the trajectories
    print(
        f"Computing predictions for all the test trajectories for {predictor.name}..."
    )

    ### Uncomment here if you want to mannualy iterate over your trajectory instead for some reason
    # all_preds = []
    # import torch
    # with torch.no_grad():
    #     for traj in tqdm(dataset.trajectories.test):  # for each test trajectory
    #         pred = torch.zeros(
    #             (traj.shape[0] - history_size, traj.shape[1], traj.shape[2])
    #         )
    #         for i in range(0, traj.shape[0] - history_size):  # for each time-step
    #             # --- this is the prediction
    #             p = predictor.predict(traj[i : i + history_size, :, :], future_size)
    #             # ---
    #             # We keep last frame of the predicted sequence (so {history_size + future_size} seconds from first input)
    #             pred[i] = p[-1]
    #         all_preds += [pred]
    # print("Predictions OK")

    # plot predicted trajectories
    print("Plotting...")
    for traj in tqdm(dataset.trajectories.test):  # for each test trajectories
        prediction, offset = predictor.predict_trajectory(traj, future_size=future_size)
        ref_traj = traj.create_subtraj(dataset.config.out_points, dataset.config.out_features)
        title = f"{predictor}_over_{ref_traj.title}"
        plot_trajs(
            [ref_traj, prediction, ref_traj],
            offsets=[0, offset, future_size],
            title=title,
            savefig_path=f"{title}.pdf",
            legend_labels=["Truth", predictor, "delayed (1s)"],
        )
