import sys
from pathlib import Path
from prescyent.evaluator import eval_predictors
from prescyent.predictor import MlpPredictor, MlpConfig, TrainingConfig
from prescyent.dataset import TeleopIcubDataset, TeleopIcubDatasetConfig
from prescyent.evaluator.plotting import plot_trajectory_prediction
from tqdm import tqdm
import torch

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage:", "plot_mlp.py data/models/teleopicub/all/MlpPredictor/version_0")

    # we load with the same config as for the training
    print("Loading the dataset...")
    dataset = TeleopIcubDataset(config=Path(sys.argv[1]) / 'dataset.config')
    history_size = dataset.config.history_size
    future_size = dataset.config.future_size
    print("Dataset OK")

    # load a pretrained model
    print("Loading the predictor...")
    predictor = MlpPredictor(sys.argv[1])
    print("Predictor OK")

    # predict the trajectories
    print(f"Computing predictions for all the test trajectories for {predictor.name}...")
    all_preds = []
    with torch.no_grad():
        for traj in tqdm(dataset.trajectories.test):
            pred = torch.zeros((traj.shape[0] - history_size, traj.shape[1], traj.shape[2]))
            for i in range(0, traj.shape[0] - history_size):
                p = predictor.get_prediction(traj[i:i+history_size, :, :], future_size)
                pred[i, :, :] = p[-1]
            all_preds += [pred]
    print("Predictions OK")
    
    # plot everything in the current directory
    print("Plotting...")
    for i in tqdm(range(len(all_preds))):
        plot_trajectory_prediction(dataset.trajectories.test[i], all_preds[i], history_size, f'plots/pred_{i}')


    # eval_results = eval_predictors([predictor],
    #                                dataset.trajectories.test[0:-1],
    #                                history_size=history_size,
    #                                future_size=future_size,
    #                                unscale_function=dataset.unscale)[0]
    # print("ADE:", eval_results.mean_ade, "FDE:", eval_results.mean_fde)
