import matplotlib as mpl
import imageio_ffmpeg

mpl.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
mpl.use("Agg")


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import math


def animate_replay(
    npz_path,
    save_path=None,
    fps=30,
):
    data = np.load(npz_path, allow_pickle=True)
    states = data["state"]      # (T, 4, 6)
    actions = data["action"]    # (T, {0,1}->{nx,nz,mu,fire})
    hp = data["hp"]             # (T, 4)

    T = states.shape[0]

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.set_aspect("equal")
    ax.set_title("2v2 Dogfight Replay (3DOF)")

    margin = 600
    ax.set_xlim(states[:, :, 0].min() - margin, states[:, :, 0].max() + margin)
    ax.set_ylim(states[:, :, 1].min() - margin, states[:, :, 1].max() + margin)

    # Aircraft markers
    A0, = ax.plot([], [], "bo", label="A0")
    A1, = ax.plot([], [], "bs", label="A1")
    B0, = ax.plot([], [], "ro", label="B0")
    B1, = ax.plot([], [], "rs", label="B1")

    trails = {
        0: ax.plot([], [], "b-", alpha=0.3)[0],
        1: ax.plot([], [], "b-", alpha=0.3)[0],
        2: ax.plot([], [], "r-", alpha=0.3)[0],
        3: ax.plot([], [], "r-", alpha=0.3)[0],
    }

    # WEZ cone lines
    wez_lines = {
        0: ax.plot([], [], "b--", alpha=0.4)[0],
        1: ax.plot([], [], "b--", alpha=0.4)[0],
    }

    # Fire markers
    fire_scatter = ax.scatter([], [], s=80, c="orange", marker="*", zorder=5)

    # HP text
    hp_text = ax.text(
        0.01, 0.99, "", transform=ax.transAxes,
        verticalalignment="top", fontsize=10
    )

    ax.legend()

    WEZ_R = 900.0
    WEZ_ANG = np.deg2rad(25)

    def draw_wez(i, frame):
        x, y, _, _, psi, _ = states[frame, i]
        th1 = psi - WEZ_ANG
        th2 = psi + WEZ_ANG
        xs = [x, x + WEZ_R * np.cos(th1), x + WEZ_R * np.cos(th2)]
        ys = [y, y + WEZ_R * np.sin(th1), y + WEZ_R * np.sin(th2)]
        wez_lines[i].set_data(xs, ys)

    def update(frame):
        xs = states[frame, :, 0]
        ys = states[frame, :, 1]

        A0.set_data([xs[0]], [ys[0]])
        A1.set_data([xs[1]], [ys[1]])
        B0.set_data([xs[2]], [ys[2]])
        B1.set_data([xs[3]], [ys[3]])

        for i in range(4):
            trails[i].set_data(states[:frame, i, 0], states[:frame, i, 1])

        # WEZ for Team A
        draw_wez(0, frame)
        draw_wez(1, frame)

        # Fire markers
        fire_x, fire_y = [], []

        for a_id in [0, 1]:
            act = actions[frame].get(a_id, None)
            if act is not None and len(act) >= 4 and act[3] > 0.5:
                fire_x.append(xs[a_id])
                fire_y.append(ys[a_id])

        if len(fire_x) > 0:
            fire_scatter.set_offsets(np.column_stack([fire_x, fire_y]))
        else:
            fire_scatter.set_offsets(np.empty((0, 2)))

        # HP display
        hp_text.set_text(
            f"HP A0:{hp[frame, 0]:.1f}  A1:{hp[frame, 1]:.1f}\n"
            f"HP B0:{hp[frame, 2]:.1f}  B1:{hp[frame, 3]:.1f}"
        )

        return A0, A1, B0, B1, fire_scatter, hp_text

    anim = FuncAnimation(
        fig, update,
        frames=T,
        interval=1000 / fps,
        blit=False
    )

    if save_path:
        writer = FFMpegWriter(
            fps=fps,
            codec="libx264",
            extra_args=[
                "-pix_fmt", "yuv420p",  # KRİTİK SATIR
                "-profile:v", "baseline"
            ]
        )
        anim.save(save_path, writer=writer)

    plt.close(fig)
