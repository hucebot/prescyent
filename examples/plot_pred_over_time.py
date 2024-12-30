"""run this script to generate an 2D animated plot of prediction vs truth over time"""

import sys
from argparse import ArgumentParser
from typing import List, Literal, Union

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm

from prescyent.dataset import Trajectory
from prescyent.predictor.base_predictor import BasePredictor
from prescyent.auto_predictor import AutoPredictor
from prescyent.auto_dataset import AutoDataset
from prescyent.utils.tensor_manipulation import trajectory_tensor_get_dim_limits


def plot_pred_over_time(
    predictor: BasePredictor,
    trajectory: Trajectory,
    future_size: int = None,
    save_dir: Union[Path, str] = "eval/video",
    file_type: Literal["mp4", "gif"] = "mp4",
    fps: int = None,
    first_rendered_frame: int = 0,
    max_rendered_frames: int = sys.maxsize,
    point_ids: List[int] = None,
):
    # check args
    if isinstance(save_dir, str):
        save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    if future_size is None:
        future_size = predictor.config.dataset_config.future_size
    history_size = predictor.config.dataset_config.history_size

    # prepare traj for input / output
    input_traj = trajectory.create_subtraj(
        predictor.config.dataset_config.in_points,
        predictor.config.dataset_config.in_features,
        predictor.config.dataset_config.context_keys,
    )
    # predict trajectory
    pred_traj, offset = predictor.predict_trajectory(input_traj, future_size)

    if point_ids is not None:
        pred_traj = pred_traj.create_subtraj(point_ids)
    else:
        point_ids = predictor.config.dataset_config.out_points

    output_traj = trajectory.create_subtraj(
        point_ids,
        predictor.config.dataset_config.out_features,
    )

    if fps is not None:  # reduce frequency of traj, + update history and future
        history_size = int(history_size / output_traj.frequency * fps)
        future_size = int(future_size / output_traj.frequency * fps)
        offset = future_size + history_size - 1
        output_traj.update_frequency(fps)
        pred_traj.update_frequency(fps)
    else:
        fps = output_traj.frequency

    # init plot
    last_rendered_frame = min(
        [len(output_traj), first_rendered_frame + max_rendered_frames]
    )
    num_points = output_traj.tensor.shape[1]
    num_dims = output_traj.tensor.shape[2]
    num_axes = num_points * num_dims

    x = np.arange(
        first_rendered_frame / output_traj.frequency,
        (last_rendered_frame + future_size) / output_traj.frequency,
        1 / output_traj.frequency,
    )

    fig, axes = plt.subplots(num_axes, sharex=True)
    if num_axes == 1:
        axes = [axes]
    truth_lines = [ax.plot([], [])[0] for ax in axes]
    pred_lines = [ax.plot([], [], linestyle="dashed")[0] for ax in axes]
    min_t, max_t = trajectory_tensor_get_dim_limits(pred_traj.tensor)
    ax_id = 0
    # set limits and labels for each axis
    for p in range(num_points):
        for d in range(num_dims):
            axes[ax_id].set_xlim(
                first_rendered_frame / output_traj.frequency, (last_rendered_frame + future_size) / output_traj.frequency
            )  # Time range
            axes[ax_id].set_ylim(min_t[d] - 0.025, max_t[d] + 0.025)  # y range
            axes[ax_id].set_ylabel(
                f"{output_traj.point_names[p]} {output_traj.tensor_features.dims_names[d]}"
            )
            ax_id += 1
    axes[-1].set_xlabel("Time (s)")
    w = min(
        (last_rendered_frame + future_size) * 0.05 + 5, 2**16 / 100 - 1
    )  # calculated values or max value accepted by matplotlib (max is 2¹⁶ pxl and default dpi is 100)
    h = min(
        len(axes) * 2 + 5, 2**16 / 100 - 1
    )  # calculated values or max value accepted by matplotlib (max is 2¹⁶ pxl and default dpi is 100)
    fig.set_size_inches(w, h)
    fig.tight_layout(pad=5)
    title = trajectory.title
    fig.suptitle(title)

    # animate function run at each frame
    def animate(i):
        ax_id = 0
        # we plot each dim of each point separately
        for p in range(num_points):
            for d in range(num_dims):
                # we plot pred moved from future_size, while its value in pred starts before offset
                # all values are according to the `first_rendered_frame` value instead of 0
                truth_x_end = min(i - first_rendered_frame + 1, len(output_traj.tensor))
                truth_start = first_rendered_frame
                truth_end = min(i + 1, len(output_traj.tensor))
                if truth_end <= len(output_traj.tensor):
                    truth_lines[ax_id].set_data(
                        x[:truth_x_end], output_traj.tensor[truth_start:truth_end, p, d]
                    )
                pred_x_start = 0
                pred_x_end = min(i - first_rendered_frame + future_size, len(pred_traj.tensor))
                pred_y_start = first_rendered_frame - offset
                pred_y_end = min(i - history_size, len(pred_traj.tensor)) + 1
                # special case when first_rendered_frame is zero,
                # we use offset to place the prediction accordingly to the truth
                if first_rendered_frame == 0:
                    pred_x_start = offset
                    pred_x_end = pred_x_start + i - history_size + 1
                    pred_y_start = 0
                    pred_y_end = i - history_size + 1
                # special case when we don't start at zero but before the first prediction
                elif pred_y_start < 0:
                    pred_x_start = - pred_y_start
                    pred_x_end = pred_x_start + i - history_size + 1
                    pred_y_start = 0
                if i >= history_size:
                    pred_lines[ax_id].set_data(
                        x[pred_x_start:pred_x_end],
                        pred_traj.tensor[pred_y_start:pred_y_end, p, d],
                    )
                ax_id += 1
        return truth_lines, pred_lines

    # create animation with function and fig
    anim = animation.FuncAnimation(
        fig,
        animate,
        frames=tqdm(
            range(first_rendered_frame, last_rendered_frame, 1),
            "Rendering trajectory frames",
            colour="red",
        ),
    )

    if file_type == "mp4":
        title = f"{trajectory.title.replace('/', '_')}_animation.mp4"
        writer = animation.FFMpegWriter(fps=fps)
        anim.save(str(save_dir / title), writer=writer)
        print(f"Saved 3d rendered prediction at {str(save_dir / title)}")
    elif file_type == "gif":
        title = f"{trajectory.title.replace('/', '_')}_animation.gif"
        writer = animation.PillowWriter(fps=fps, bitrate=1800)
        anim.save(str(save_dir / title), writer=writer)
        print(f"Saved 3d rendered prediction at {str(save_dir / title)}")
    else:
        raise AttributeError(
            f'file_type can be "mp4" or "gif" not "{file_type}"'
        )


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

    # we load dataset with the same config as for the training
    print("Loading the dataset...")
    dataset = AutoDataset.build_from_config(path)
    print("Dataset loaded !")

    # load a pretrained model
    print("Loading the predictor...")
    predictor = AutoPredictor.load_pretrained(path)
    predictor.describe()
    print("Predictor loaded !")

    #? test a baseline here !
    # from prescyent.predictor import ConstantPredictor, PredictorConfig
    # predictor = ConstantPredictor(PredictorConfig(dataset_config=dataset.config))

    test_traj = dataset.trajectories.test[2]
    plot_pred_over_time(
        predictor=predictor,
        trajectory=test_traj,
        # BEWARE THAT `future_size` IS USED FOR AT PREDICTOR LEVEL BEFORE THE FPS REDUCTION FOR PLOTTING
        # ALSO IT CANNOT BE USED WITH ALL PREDICTORS, AS IT MAY REQUIRE TO LOOP OVER PREDICTION TO PREDICT LONGER FUTURE
        # future_size=int(test_traj.frequency * 3),
        save_dir="video_plot",
        file_type="gif",
        # first_rendered_frame=dataset.config.history_size + dataset.config.future_size - 1,  # start at first pred frame
        fps=10,  # Plot a subsample of the prediction at 10Hz
        max_rendered_frames=100,  # output only 10 seconds of prediction
        point_ids=[-1],  # plot only last point's features
    )
