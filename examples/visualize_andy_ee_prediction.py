"""
This script is used to generate the visualization you can see in the README
Here we plot only the coordinates of the trajectory along with the last predicted frame at T + future_size,
    which sees its opacity decay until disappearing as it arrives at frame T
The plot generated in the readme was from a model trained in the `benchmark/andydataset`,
    predicting coordinates and rotation for only the right hand based on all joints history
After training a predictor for AndyDataset, or another 3D dataset,
    you can come back to this script as some reference to plot or run your predictor
"""

from argparse import ArgumentParser
from pathlib import Path
import sys
from typing import Literal, Union

import matplotlib
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import tqdm

from prescyent.auto_predictor import AutoPredictor
from prescyent.auto_dataset import AutoDataset
from prescyent.predictor.base_predictor import BasePredictor
from prescyent.dataset import Trajectory
from prescyent.utils.logger import logger, EVAL
from prescyent.utils.tensor_manipulation import trajectory_tensor_get_dim_limits

# Constants for the plot

ELEVATION = 10.0
MIN_MAX_EPS = 0.1

POINT_COLOR = "#2934CC"
POINT_SHAPE = "o"
BONES_COLOR = "#2986CC"

PRED_POINT_COLOR = "#CC2828"
PRED_POINT_SHAPE = "s"
PRED_BONES_COLOR = "#ff3333"


def render_prediction(
    trajectory: Trajectory,
    predictor: BasePredictor,
    future_size: int,
    save_file_format: Literal["mp4", "gif", None] = None,
    save_dir: Union[str, Path] = "data/eval/visualizations",
    min_max_layout: bool = True,
    interactive: bool = True,
    turn_view: bool = True,
    first_rendered_frames: int = 0,
    max_rendered_frames: int = sys.maxsize,
):
    """runs prediction and renders the results along with the trajectory in 3D
    THIS FUNCTION HAS THE ASSUMPTION THAT YOUR TRAJECTORY'S FEATURES ARE [CoordinatesXYZ([0, 1, 2])

    Args:
        trajectory (Trajectory): the trajectory we predict over
        predictor (BasePredictor): the predictor that we'll use over the trajectory
        future_size (int): the T+F future frames we want to predict.
                Note that we'll always plot the latest predicted frame,
                also beware that some model cannot output a future greater that the one they were trained for !!
        save_file_format (Literal["mp4", "gif"], optional): output gif or mp4. If None, runs only interactively. Defaults to None.
        save_dir (str): directory where the plot will be saved
        min_max_layout (bool, optional): if True sets coordinates limits to the min_max of the traj tensors. Defaults to True.
        interactive (bool, optional): if true, render interactively. Defaults to True.
        turn_view (bool, optional): if true, the view turns on each frame around the target. Defaults to True.
        first_rendered_frames (int, optional): id of the first rendered frame. Defaults to 0.
        max_rendered_frames (int, optional): id of the last rendered frame. Defaults to sys.maxsize.

    Raises:
        AttributeError: tried to draw more than 5 trajectories (convenient error, feel free to adapt the method)
        AttributeError: save_file_format is not supported
    """
    # check that the savedir exist if it is needed
    if not isinstance(save_dir, Path):
        save_dir = Path(save_dir)
    if save_file_format and not Path(save_dir).exists():
        save_dir.mkdir(parents=True, exist_ok=True)
    # init the plot
    test_frame = trajectory.tensor[0].transpose(0, 1).tolist()
    input_traj = trajectory.create_subtraj(
        predictor.config.dataset_config.in_points,
        predictor.config.dataset_config.in_features,
        predictor.config.dataset_config.context_keys,
    )
    plt.rcParams["figure.figsize"] = [12.20, 10.80]
    plt.rcParams["figure.autolayout"] = True
    fig = plt.figure()
    ax_3d = fig.add_subplot(projection="3d", proj_type="ortho")
    if min_max_layout:
        min_t, max_t = trajectory_tensor_get_dim_limits(trajectory.tensor)
        ax_3d.set_xlim3d([min_t[0] - MIN_MAX_EPS, max_t[0] + MIN_MAX_EPS])
        ax_3d.set_ylim3d([min_t[1] - MIN_MAX_EPS, max_t[1] + MIN_MAX_EPS])
        ax_3d.set_zlim3d([min_t[2] - MIN_MAX_EPS, max_t[2] + MIN_MAX_EPS])
    try:
        ax_3d.set_aspect("equal", adjustable="box")
    except NotImplementedError:
        ax_3d.set_aspect("auto")
    ax_3d.set_title(str(trajectory.title))
    bone_list = [
        ax_3d.plot([], [], [], color=BONES_COLOR, zdir="z")
        for _ in range(len(test_frame[0]))
    ]
    point_list = ax_3d.plot([], [], [], POINT_SHAPE, c=POINT_COLOR, zdir="z", ms=6)
    pred_point_list = [
        ax_3d.plot(
            [],
            [],
            [],
            PRED_POINT_SHAPE,
            c=PRED_POINT_COLOR,
            zdir="z",
            ms=6,
            alpha=f / future_size,
        )
        for f in range(future_size)
    ]
    pred_bone_list = [
        [
            ax_3d.plot([], [], [], color=PRED_BONES_COLOR, zdir="z")
            for _ in range(len(test_frame[0]))
        ]
        for _ in range(future_size)
    ]
    for axis in [ax_3d.xaxis, ax_3d.yaxis, ax_3d.zaxis]:
        axlabel = axis.axis_name
        axis.set_label_text(axlabel)
    ax_3d.view_init(elev=ELEVATION, azim=315)

    def init():
        """sets default value of all point and bone lists"""
        for point_id, _ in enumerate((test_frame[0])):
            bone_list[point_id][0].set_data([], [])
            bone_list[point_id][0].set_3d_properties([])
        point_list[0].set_data([], [])
        point_list[0].set_3d_properties([])
        for f in range(future_size):
            pred_point_list[f][0].set_data([], [])
            pred_point_list[f][0].set_3d_properties([])
            for point_id, _ in enumerate((test_frame[0])):
                pred_bone_list[f][point_id][0].set_data([], [])
                pred_bone_list[f][point_id][0].set_3d_properties([])
        return (
            point_list,
            bone_list,
            pred_point_list,
            pred_bone_list,
        )

    # animate function updates the plot
    def animate(i):
        """function in charge of updating each points and bones at a given timestep
        here we plot the trajectory at frame i and prediction at frame i+future_size + update the previous predictions states to keep a trail
        """
        history_size = predictor.config.dataset_config.history_size
        # plot the trajectory at time i
        if i >= 0:
            frame = trajectory.tensor[i]
            frame = frame.transpose(0, 1)
            frame = frame.tolist()
            xs = frame[0]
            ys = frame[1]
            zs = frame[2]
            point_list[0].set_data(xs, ys)
            point_list[0].set_3d_properties(zs)
            for point, _ in enumerate(frame[0]):
                if trajectory.point_parents[point] != -1:
                    bxs = [frame[0][point], frame[0][trajectory.point_parents[point]]]
                    bys = [frame[1][point], frame[1][trajectory.point_parents[point]]]
                    bzs = [frame[2][point], frame[2][trajectory.point_parents[point]]]
                    bone_list[point][0].set_data(bxs, bys)
                    bone_list[point][0].set_3d_properties(bzs)
        # if we have enough history to predict
        if i >= history_size:
            # we get prediction from predictor
            input_tensor = input_traj.tensor[i - history_size : i]
            context = {
                c_key: c_value[i - history_size : i]
                for c_key, c_value in input_traj.context.items()
            }
            pred_tensor = predictor.predict(
                input_tensor, future_size=future_size, context=context
            )
            # move all previous predicted bone list and point list of one timestep
            for f in range(future_size - 1):
                x_next, y_next, z_next = pred_point_list[f + 1][0].get_data_3d()
                # update the current point with the next point's data
                pred_point_list[f][0].set_data(x_next, y_next)
                pred_point_list[f][0].set_3d_properties(z_next)
                for point, _ in enumerate(frame[0]):
                    if trajectory.point_parents[point] != -1:
                        bx_next, by_next, bz_next = pred_bone_list[f + 1][point][
                            0
                        ].get_data_3d()
                        pred_bone_list[f][point][0].set_data(bx_next, by_next)
                        pred_bone_list[f][point][0].set_3d_properties(bz_next)
            # update last pred bones and points
            last_frame = pred_tensor[-1]
            frame = last_frame.transpose(0, 1)
            frame = frame.tolist()
            xs = frame[0]
            ys = frame[1]
            zs = frame[2]
            pred_point_list[future_size - 1][0].set_data(xs, ys)
            pred_point_list[future_size - 1][0].set_3d_properties(zs)
            for point, _ in enumerate(frame[0]):
                if trajectory.point_parents[point] != -1:
                    bxs = [frame[0][point], frame[0][trajectory.point_parents[point]]]
                    bys = [frame[1][point], frame[1][trajectory.point_parents[point]]]
                    bzs = [frame[2][point], frame[2][trajectory.point_parents[point]]]
                    pred_bone_list[future_size - 1][point][0].set_data(bxs, bys)
                    pred_bone_list[future_size - 1][point][0].set_3d_properties(bzs)
        if not interactive and turn_view:
            ax_3d.view_init(
                elev=ELEVATION, azim=int(i * 10 / trajectory.frequency) % 360
            )
        ax_3d.legend(
            handles=[point_list[0], pred_point_list[0][0]],
            labels=[trajectory.title, str(predictor)],
        )
        return (
            point_list,
            bone_list,
            pred_point_list,
            pred_bone_list,
        )

    anim = FuncAnimation(
        fig,
        animate,
        init_func=init,
        frames=tqdm(
            range(
                first_rendered_frames,
                min([len(trajectory), first_rendered_frames + max_rendered_frames]),
            ),
            "Rendering trajectory frames",
            colour="red",
        ),
    )

    if not save_file_format:
        pass
    elif save_file_format == "mp4":
        title = f"{trajectory.title.replace('/', '_')}_animation.mp4"
        anim.save(
            str(save_dir / title),
            fps=trajectory.frequency,
            extra_args=["-vcodec", "libx264"],
        )
        logger.getChild(EVAL).info(
            f"Saved 3d rendered prediction at {str(save_dir / title)}"
        )
    elif save_file_format == "gif":
        title = f"{trajectory}_animation.gif"
        writer = matplotlib.animation.PillowWriter(
            fps=trajectory.frequency, bitrate=1800
        )
        anim.save(str(save_dir / title), writer=writer)
        logger.getChild(EVAL).info(
            f"Saved 3d rendered prediction at {str(save_dir / title)}"
        )
    else:
        raise AttributeError(
            f'save_file_format can be "mp4", "gif" or None, not "{save_file_format}"'
        )
    if interactive:
        try:
            matplotlib.use("TkAgg")
        except ImportError:
            logger.getChild(EVAL).warning("can't use TkAgg backend for matplotlib")
            matplotlib.use("agg")
        plt.show()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("model_path", help="data/models/Andy_ee/MlpPredictor/version_0")
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

    render_prediction(
        dataset.trajectories.test[0],
        predictor,
        dataset.config.future_size,
        save_file_format="mp4",
        first_rendered_frames=120,  # first frame start at  T = 5 seconds (dataset is at 24Hz)
        # max_rendered_frames=240,  # duration of 10 seconds (dataset is at 24Hz)
    )
