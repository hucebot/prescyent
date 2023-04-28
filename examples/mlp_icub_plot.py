import glob, os, sys
from tqdm import tqdm
import torch
from pathlib import Path
from prescyent.evaluator import eval_predictors
from prescyent.predictor import MlpPredictor, MlpConfig, TrainingConfig
from prescyent.dataset import TeleopIcubDataset, TeleopIcubDatasetConfig
from prescyent.evaluator.plotting import plot_trajs

if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] == '--help':
        print("Usage (examples):")
        print("plot_mlp.py --model data/models/teleopicub/all/MlpPredictor/version_0")
        print("plot_mlp.py --last data/models/teleopicub/all/MlpPredictor/")
        # sys.exit(1)
        path = "data/models/teleopicub/all/MlpPredictor/version_2/"

    elif sys.argv[1] == '--model':
        path = sys.argv[2]
    elif sys.argv[1] == '--last':
        path =  max(glob.glob(sys.argv[2] + '/version_*'), key=os.path.getctime)
    else:
        path = "data/models/teleopicub/all/MlpPredictor/version_2/config.json"
        print("Error: use --last or --model")

    print("Path:", path)
    # we load with the same config as for the training
    print("Loading the dataset...")
    dataset = TeleopIcubDataset(path + '/dataset.config')
    history_size = dataset.config.history_size
    future_size = dataset.config.future_size
    print("Dataset OK")

    # load a pretrained model
    print("Loading the predictor...")
    predictor = MlpPredictor(path)
    print("Predictor OK")

    # predict the trajectories
    print(f"Computing predictions for all the test trajectories for {predictor.name}...")
    all_preds = []
    with torch.no_grad():
        for traj in tqdm(dataset.trajectories.test): # for each test trajectories
            pred = torch.zeros((traj.shape[0] - history_size, traj.shape[1], traj.shape[2]))
            for i in range(0, traj.shape[0] - history_size): # for each time-step
                ##### this is the prediction
                p = predictor.predict(traj[i:i+history_size, :, :], future_size)
                ######
                pred[i, :, :] = p[-1]
            all_preds += [pred]
    print("Predictions OK")

    # plot everything in the current directory
    print("Plotting...")
    for i in tqdm(range(len(all_preds))): # for each test trajectory
        ref_traj = dataset.trajectories.test[i]
        title = '/'.join(ref_traj.file_path.parts[-2:])
        dims = ref_traj.dimension_names
        plot_trajs([ref_traj.tensor, all_preds[i], ref_traj.tensor], 
                   f'plots/pred_{i}.pdf', 
                   # we shift the predictions of the history_size (data we need to predict) and of future_size (because given 1 s of data, we predict 1 second from now)
                   shifts=[0, history_size + future_size, future_size],
                   group_labels=dims,
                   dim_labels = ['x', 'y', 'z'],
                   title=title,
                   traj_labels=['Truth', 'Prediction', 'delayed (1s)'])