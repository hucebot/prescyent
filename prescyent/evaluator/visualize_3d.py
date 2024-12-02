import numpy as np
import sys
from pathlib import Path
from tqdm import tqdm
from typing import List, Literal

import matplotlib
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

from prescyent.dataset import Trajectory
from prescyent.dataset.features import Rotation
from prescyent.utils.logger import logger, EVAL
from prescyent.utils.tensor_manipulation import trajectory_tensor_get_dim_limits


POINT_COLORS = ["k", "b", "g", "r", "c"]
POINT_SHAPES = ["o", "s", "v", "*", "D"]
BONES_COLORS = ["#0f0f0f80", "#0f0f0f80", "#0f0f0f80", "#0f0f0f80", "#0f0f0f80"]


def plot_rotated_axes(ax, r, name=None, offset=(0, 0, 0), scale=0.05):
    """from scipy doc here:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html
    """
    colors = ("#FF6666", "#005533", "#1199EE")  # Colorblind-safe RGB
    loc = np.array([offset, offset])
    for i, c in enumerate(colors):
        line = np.zeros((2, 3))
        line[1, i] = scale
        line_rot = r.apply(line)
        line_plot = line_rot + loc
        rotation = ax.plot(line_plot[:, 0], line_plot[:, 1], line_plot[:, 2], c)
        return rotation


def render_3d_trajectories(
    trajs: List[Trajectory],
    offsets: List[int],
    save_file_format: Literal["mp4", "gif", None] = None,
    save_dir: str = "data/eval/visualizations",
    min_max_layout: bool = True,
    interactive: bool = True,
    draw_bones: bool = True,
    turn_view: bool = True,
    first_rendered_frames: int = 0,
    max_rendered_frames: int = sys.maxsize,
):
    """render a 3D plot of a list of trajectories

    Args:
        trajs (List[Trajectory]): list of trajectories
        offsets (List[int]): list of offsets
        save_file_format (Literal["mp4", "gif"], optional): output gif or mp4. If None, runs only interactively. Defaults to None.
        min_max_layout (bool, optional): if True sets coordinates limits to the min_max of the traj tensors. Defaults to True.
        interactive (bool, optional): if true, render interactively. Defaults to True.
        draw_bones (bool, optional): if true, draw segments between each points, using traj.point_parents infos. Defaults to True.
        turn_view (bool, optional): if true, the view turns on each frame around the target. Defaults to True.
        first_rendered_frames (int, optional): id of the first rendered frame. Defaults to 0.
        max_rendered_frames (int, optional): id of the last rendered frame. Defaults to sys.maxsize.

    Raises:
        AttributeError: tried to draw more than 5 trajectories (convenient error, feel free to adapt the method)
        AttributeError: save_file_format is not supported
    """

    if len(trajs) >= 6:
        raise AttributeError(
            "We cannot draw more than 5 trajectories at a time (if you want to, remove this error and add some more options to POINT_COLORS, POINT_SHAPES and BONES_COLORS)"
        )
    if isinstance(save_dir, str) and save_file_format is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    if save_file_format and not Path(save_dir).exists():
        Path(save_dir).mkdir(exist_ok=True)
    elevation = 20.0
    rot_colors = ("#FF6666", "#005533", "#1199EE")  # Colorblind-safe RGB
    test_frames = [traj.tensor[0].transpose(0, 1).tolist() for traj in trajs]
    plt.rcParams["figure.figsize"] = [12.20, 10.80]
    plt.rcParams["figure.autolayout"] = True
    fig = plt.figure()
    ax_3d = fig.add_subplot(projection="3d", proj_type="ortho")
    if min_max_layout:
        min_t, max_t = trajectory_tensor_get_dim_limits(trajs[0].tensor)
        ax_3d.set_xlim3d([min_t[0], max_t[0]])
        ax_3d.set_ylim3d([min_t[1], max_t[1]])
        ax_3d.set_zlim3d([min_t[2], max_t[2]])
    try:
        ax_3d.set_aspect("equal", adjustable="box")
    except NotImplementedError:
        ax_3d.set_aspect("auto")
    ax_3d.set_title(str(trajs[0].title))
    rotation_list = []
    draw_rotation = all(
        [
            any([isinstance(feat, Rotation) for feat in traj.tensor_features])
            for traj in trajs
        ]
    )
    if draw_rotation:
        rotation_scale = 0.05
        rotation_list = [
            [
                [ax_3d.plot([], [], [], c) for c in rot_colors]
                for _ in range(len(test_frame[0]))
            ]
            for test_frame in test_frames
        ]
    if draw_bones:
        # assert len(traj.point_parents) == len(
        #     test_frame[0]
        # )  # assert that we have parent for each point
        bone_list = [
            [
                ax_3d.plot([], [], [], color=BONES_COLORS[f], zdir="z")
                for _ in range(len(test_frame[0]))
            ]
            for f, test_frame in enumerate(test_frames)
        ]
    point_list = [
        ax_3d.plot([], [], [], POINT_SHAPES[f], c=POINT_COLORS[f], zdir="z", ms=6)
        for f, _ in enumerate(test_frames)
    ]
    for _, (axis, c) in enumerate(
        zip((ax_3d.xaxis, ax_3d.yaxis, ax_3d.zaxis), rot_colors)
    ):
        axlabel = axis.axis_name
        axis.set_label_text(axlabel)
        axis.label.set_color(c)
        axis.line.set_color(c)
        axis.set_tick_params(colors=c)
    ax_3d.view_init(elev=elevation, azim=315)

    def init():
        for t, traj_point_list in enumerate(point_list):
            if draw_bones:
                for point, _ in enumerate(traj_point_list):
                    bone_list[t][point][0].set_data([], [])
                    bone_list[t][point][0].set_3d_properties([])
            traj_point_list[0].set_data([], [])
            traj_point_list[0].set_3d_properties([])
        return (
            point_list,
            bone_list,
            rotation_list,
        )

    def animate(i):
        for traj_id, (offset, traj) in enumerate(zip(offsets, trajs)):
            frame_id = i - offset
            if frame_id >= 0:
                frame = traj.tensor[frame_id]
                frame = frame.transpose(0, 1)
                frame = frame.tolist()
                xs = frame[0]
                ys = frame[1]
                zs = frame[2]
                point_list[traj_id][0].set_data(xs, ys)
                point_list[traj_id][0].set_3d_properties(zs)
                if draw_bones:
                    for point, _ in enumerate(frame[0]):
                        if traj.point_parents[point] != -1:
                            bxs = [frame[0][point], frame[0][traj.point_parents[point]]]
                            bys = [frame[1][point], frame[1][traj.point_parents[point]]]
                            bzs = [frame[2][point], frame[2][traj.point_parents[point]]]
                            bone_list[traj_id][point][0].set_data(bxs, bys)
                            bone_list[traj_id][point][0].set_3d_properties(bzs)
                if draw_rotation:
                    for point, x in enumerate(xs):
                        offset = (x, ys[point], zs[point])
                        loc = np.array([offset, offset])
                        rotation = traj.get_scipy_rotation(i, point)
                        for col, _ in enumerate(rot_colors):
                            line = np.zeros((2, 3))
                            line[1, col] = rotation_scale
                            line_rot = rotation.apply(line)
                            line_plot = line_rot + loc
                            rotation_list[traj_id][point][col][0].set_data(
                                line_plot[:, 0], line_plot[:, 1]
                            )
                            rotation_list[traj_id][point][col][0].set_3d_properties(
                                line_plot[:, 2]
                            )
        if not interactive and turn_view:
            ax_3d.view_init(elev=elevation, azim=int(i * 10 / trajs[0].frequency) % 360)
        ax_3d.legend(
            handles=[point_list[traj_id][0] for traj_id, _ in enumerate(trajs)],
            labels=[traj.title for traj in trajs],
            labelcolor=[POINT_COLORS[f] for f, _ in enumerate(trajs)],
        )
        return (
            point_list,
            bone_list,
            rotation_list,
        )

    anim = FuncAnimation(
        fig,
        animate,
        init_func=init,
        frames=tqdm(
            range(
                first_rendered_frames,
                min(
                    [len(traj) for traj in trajs]
                    + [first_rendered_frames + max_rendered_frames]
                ),
            ),
            "Rendering trajectory frames",
            colour="red",
        ),
    )

    if not save_file_format:
        pass
    elif save_file_format == "mp4":
        title = f"{trajs[0].title.replace('/', '_')}_animation.mp4"
        anim.save(
            str(save_dir / title),
            fps=trajs[0].frequency,
            extra_args=["-vcodec", "libx264"],
        )
        logger.getChild(EVAL).info(
            f"Saved 3d rendered prediction at {str(save_dir / title)}"
        )
    elif save_file_format == "gif":
        title = f"{trajs[0]}_animation.gif"
        writer = matplotlib.animation.PillowWriter(fps=trajs[0].frequency, bitrate=1800)
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
