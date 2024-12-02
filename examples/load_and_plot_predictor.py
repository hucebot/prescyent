# Script to run evaluation over a predictor loaded from a path.
# Evaluation is performed on the test set of the dataset used to train the predictor
# We load everything from the predictor's config path.
from argparse import ArgumentParser
from pathlib import Path

from tqdm import tqdm

from prescyent.auto_predictor import AutoPredictor
from prescyent.auto_dataset import AutoDataset
from prescyent.evaluator.plotting import plot_trajs, plot_mpjpe


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "model_path", help="data/models/teleopicub/all/MlpPredictor/version_0"
    )
    args = parser.parse_args()
    path = Path(args.model_path)
    if not path.is_dir():
        path = path.parent
    print("Path:", path)
    # we load with the same config as for the training
    print("Loading the dataset...")
    dataset = AutoDataset.build_from_config(path)
    print("Dataset loaded !")

    # load a pretrained model
    print("Loading the predictor...")
    predictor = AutoPredictor.load_pretrained(path)
    predictor.describe()
    print("Predictor loaded !")

    # predict the trajectories
    print(
        f"Computing predictions for all the test trajectories for {predictor.name}..."
    )
    history_size = dataset.config.history_size
    future_size = dataset.config.future_size

    # # Uncomment this section if you want to manually iterate over your trajectory instead for some reason
    # all_preds = []
    # import
    # with .no_grad():
    #     for traj in tqdm(dataset.trajectories.test):  # for each test trajectory
    #         pred = .zeros(
    #             (traj.shape[0] - history_size, traj.shape[1], traj.shape[2])
    #         )
    #         for i in range(0, traj.shape[0] - history_size):  # for each time-step
    #             # --- this is the prediction
    #             context = None
    #             if traj.context:
    #                 context = {c_name: c_tensor[i : i + history_size] for c_name, c_tensor in traj.context.items()}
    #             p = predictor.predict(traj[i : i + history_size, :, :], future_size, context)
    #             # ---
    #             # We keep last frame of the predicted sequence (so {history_size + future_size} seconds from first input)
    #             pred[i] = p[-1]
    #         all_preds += [pred]
    # print("Predictions OK")
    # # Here you get the list of the predicted tensor !

    print("Plotting...")
    # plot MPJPE evaluation metric for this predictor
    plot_mpjpe(predictor, dataset, savefig_dir_path=path / "test_plots")
    # plot predicted trajectories
    for test_traj in tqdm(
        dataset.trajectories.test,
        desc="Iterate over test trajectories",
        colour="green",
    ):  # for each test trajectories
        # We prepare our trajectory for input if a transformation is needed
        input_traj = test_traj.create_subtraj(
            dataset.config.in_points,
            dataset.config.in_features,
            dataset.config.context_keys,
        )
        # we create a new predicted trajectory from a given predictor
        predicted_traj, offset = predictor.predict_trajectory(
            input_traj, future_size=future_size
        )
        # subsample the truth trajectory if needed to compare with prediction
        truth_traj = test_traj.create_subtraj(
            dataset.config.out_points, dataset.config.out_features
        )
        title = f"{predictor}_over_{truth_traj.title}"
        # plot prediction along truth and delayed truth
        plot_trajs(
            [truth_traj, predicted_traj, truth_traj],
            offsets=[0, offset, future_size],
            title=title,
            savefig_path=path / f"test_plots/{title}.pdf",
            legend_labels=[
                "Truth",
                predictor,
                f"Delayed ({future_size/truth_traj.frequency}s)",
            ],
        )
