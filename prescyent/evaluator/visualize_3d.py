from tqdm import tqdm
from typing import List

import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation

from prescyent.dataset.trajectories import Trajectory
from prescyent.utils.logger import logger, EVAL


def render_3d_trajectory(
    traj: Trajectory,
    radius: int = 2,
    save_mp4: bool = True,
    interactive: bool = False,
    draw_bones: bool = True,
):
    """"""
    test_frame = traj[0].transpose(
        0, 1
    )  # at frame n we have (points, dims) => (dims, points)
    test_frame = test_frame.tolist()
    if len(test_frame) > 3 or len(test_frame) < 2:
        raise AttributeError(
            f"Trajectory with {len(test_frame)} dimensions can't be 3d plotted"
        )

    plt.rcParams["figure.figsize"] = [19.20, 10.80]
    plt.rcParams["figure.autolayout"] = True
    fig = plt.figure()
    ax_3d = plt.axes(projection="3d")
    ax_3d.view_init(elev=15.0)
    ax_3d.set_xlim3d([-radius / 2, radius / 2])
    ax_3d.set_zlim3d([-radius / 2, radius / 2])
    ax_3d.set_ylim3d([-radius / 2, radius / 2])
    try:
        ax_3d.set_aspect("equal")
    except NotImplementedError:
        ax_3d.set_aspect("auto")
    ax_3d.set_title(str(traj))

    if draw_bones:
        assert len(traj.point_parents) == len(
            test_frame[0]
        )  # assert that we have parent for each point
        bone_list = [
            ax_3d.plot([], [], [], color="red", zdir="z")
            for _ in range(len(test_frame[0]))
        ]
    point_list = ax_3d.plot([], [], [], "o", c="k", zdir="z", ms=6)

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
        )

    def animate(i):
        frame = traj[i]
        frame = frame.transpose(0, 1)
        frame = frame.tolist()
        xs = frame[0]
        ys = frame[1]
        zs = frame[2]
        point_list[0].set_data(xs, ys)
        point_list[0].set_3d_properties(zs)
        if draw_bones:
            xs, ys, zs = [], [], []
            for point, _ in enumerate(frame[0]):
                if traj.point_parents[point] != -1:
                    xs = [frame[0][point], frame[0][traj.point_parents[point]]]
                    ys = [frame[1][point], frame[1][traj.point_parents[point]]]
                    zs = [frame[2][point], frame[2][traj.point_parents[point]]]
                    bone_list[point][0].set_data(xs, ys)
                    bone_list[point][0].set_3d_properties(zs)
        return (
            point_list,
            bone_list,
        )

    anim = FuncAnimation(
        fig,
        animate,
        init_func=init,
        frames=tqdm(range(len(traj)), "Rendering trajectory frames"),
    )

    if save_mp4:
        title = f"{traj}_animation.mp4"
        anim.save(title, fps=traj.frequency, extra_args=["-vcodec", "libx264"])
        logger.getChild(EVAL).info(f"Saved 3d rendered prediction at {title}")
    if interactive:
        try:
            matplotlib.use("GTK3Agg")
        except AttributeError:
            print("can't use GTK3Agg backend for matplotlib")
            matplotlib.use("agg")
        plt.show()
