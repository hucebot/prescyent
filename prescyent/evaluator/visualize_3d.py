import numpy as np
from tqdm import tqdm
from typing import List

import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation

from prescyent.dataset import Trajectory
from prescyent.utils.logger import logger, EVAL
from prescyent.utils.tensor_manipulation import trajectory_tensor_get_dim_limits


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


def render_3d_trajectory(
    traj: Trajectory,
    save_file: str = None,  # use "mp4" or "gif"
    min_max_layout: bool = True,
    interactive: bool = True,
    draw_bones: bool = True,
    draw_rotation: bool = False,
    turn_view: bool = True,
):
    """"""
    elevation = 20.0
    rot_colors = ("#FF6666", "#005533", "#1199EE")  # Colorblind-safe RGB
    test_frame = traj[0].transpose(
        0, 1
    )  # at frame n we have (points, dims) => (dims, points)
    test_frame = test_frame.tolist()
    plt.rcParams["figure.figsize"] = [12.20, 10.80]
    plt.rcParams["figure.autolayout"] = True
    fig = plt.figure()
    ax_3d = fig.add_subplot(projection="3d", proj_type="ortho")
    if min_max_layout:
        min_t, max_t = trajectory_tensor_get_dim_limits(traj.tensor)
        ax_3d.set_xlim3d([min_t[0], max_t[0]])
        ax_3d.set_ylim3d([min_t[1], max_t[1]])
        ax_3d.set_zlim3d([min_t[2], max_t[2]])
    try:
        ax_3d.set_aspect("equal", adjustable="box")
    except NotImplementedError:
        ax_3d.set_aspect("auto")
    ax_3d.set_title(str(traj))
    if draw_rotation:
        rotation_scale = 0.025
        rotation_list = [
            [ax_3d.plot([], [], [], c) for c in rot_colors]
            for _ in range(len(test_frame[0]))
        ]
    if draw_bones:
        assert len(traj.point_parents) == len(
            test_frame[0]
        )  # assert that we have parent for each point
        bone_list = [
            ax_3d.plot([], [], [], color="red", zdir="z")
            for _ in range(len(test_frame[0]))
        ]
    point_list = ax_3d.plot([], [], [], "o", c="k", zdir="z", ms=6)
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
        if draw_bones:
            for point, _ in enumerate(point_list):
                bone_list[point][0].set_data([], [])
                bone_list[point][0].set_3d_properties([])
        point_list[0].set_data([], [])
        point_list[0].set_3d_properties([])
        return (
            point_list,
            bone_list,
            rotation_list,
        )

    def animate(i):
        frame = traj.tensor[i]
        frame = frame.transpose(0, 1)
        frame = frame.tolist()
        xs = frame[0]
        ys = frame[1]
        zs = frame[2]
        point_list[0].set_data(xs, ys)
        point_list[0].set_3d_properties(zs)
        if draw_bones:
            for point, _ in enumerate(frame[0]):
                if traj.point_parents[point] != -1:
                    bxs = [frame[0][point], frame[0][traj.point_parents[point]]]
                    bys = [frame[1][point], frame[1][traj.point_parents[point]]]
                    bzs = [frame[2][point], frame[2][traj.point_parents[point]]]
                    bone_list[point][0].set_data(bxs, bys)
                    bone_list[point][0].set_3d_properties(bzs)
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
                    rotation_list[point][col][0].set_data(
                        line_plot[:, 0], line_plot[:, 1]
                    )
                    rotation_list[point][col][0].set_3d_properties(line_plot[:, 2])
        if not interactive and turn_view:
            ax_3d.view_init(elev=elevation, azim=i % 360)
        return (
            point_list,
            bone_list,
            rotation_list,
        )

    anim = FuncAnimation(
        fig,
        animate,
        init_func=init,
        frames=tqdm(range(len(traj)), "Rendering trajectory frames"),
    )

    if not save_file:
        pass
    elif save_file == "mp4":
        title = f"{traj}_animation.mp4"
        anim.save(title, fps=traj.frequency, extra_args=["-vcodec", "libx264"])
        logger.getChild(EVAL).info(f"Saved 3d rendered prediction at {title}")
    elif save_file == "gif":
        title = f"{traj}_animation.gif"
        writer = matplotlib.animation.PillowWriter(fps=traj.frequency, bitrate=1800)
        anim.save(title, writer=writer)
    else:
        raise AttributeError(
            f'save_file can be "mp4", "gif" or None, not "{save_file}"'
        )
    if interactive:
        try:
            matplotlib.use("TkAgg")
        except AttributeError:
            print("can't use TkAgg backend for matplotlib")
            matplotlib.use("agg")
        plt.show()
