import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter

from scenarios.scenario_1v1.dogfight_wrappers import heuristic_policy_continuous


# ======================================================
# LEAD POINT (VISUAL ONLY)
# ======================================================
def _compute_lead_point(shooter_state, target_state, bullet_speed=600.0):
    xs, ys, _, _, _ = shooter_state
    xt, yt, vt, psit, _ = target_state

    vtx = vt * np.cos(psit)
    vty = vt * np.sin(psit)

    dx = xt - xs
    dy = yt - ys
    R = np.hypot(dx, dy) + 1e-6

    vrad = (dx * vtx + dy * vty) / R
    denom = bullet_speed - vrad
    t_hit = R / bullet_speed if denom <= 1e-3 else R / denom
    t_hit = np.clip(t_hit, 0.0, 10.0)

    return xt + vtx * t_hit, yt + vty * t_hit


# ======================================================
# FRAME DRAW
# ======================================================
def _frame(ax, env_base, info, shots, flashes, trail0, trail1,
           arena=4000.0, wez_R=800.0, wez_ang=np.deg2rad(25)):

    ax.clear()
    ax.set_aspect("equal", "box")
    ax.set_xlim(-arena, arena)
    ax.set_ylim(-arena, arena)
    ax.grid(True, alpha=0.2)

    s = env_base.s
    x0, y0, v0, psi0, _ = s[0]
    x1, y1, v1, psi1, _ = s[1]

    ax.add_artist(plt.Circle((0, 0), arena, fill=False, ls="--", alpha=0.3))

    if len(trail0) > 1:
        xs, ys = zip(*trail0)
        ax.plot(xs, ys, color="tab:blue", lw=1.0, alpha=0.5)
    if len(trail1) > 1:
        xs, ys = zip(*trail1)
        ax.plot(xs, ys, color="tab:red", lw=1.0, alpha=0.5)

    def draw_aircraft(x, y, v, psi, color):
        L = 100.0
        ax.plot([x, x + L * np.cos(psi)], [y, y + L * np.sin(psi)],
                color=color, lw=2.0)
        ax.add_artist(plt.Circle((x, y), 25, color=color, alpha=0.8))
        ax.arrow(
            x, y,
            1.5 * v * np.cos(psi),
            1.5 * v * np.sin(psi),
            head_width=70, head_length=120,
            color=color, alpha=0.6, lw=1.5,
            length_includes_head=True
        )

    draw_aircraft(x0, y0, v0, psi0, "tab:blue")
    draw_aircraft(x1, y1, v1, psi1, "tab:red")

    ax.plot([x0, x1], [y0, y1], color="gray", alpha=0.4)

    theta = np.linspace(-wez_ang, wez_ang, 60) + psi0
    ax.plot(x0 + wez_R * np.cos(theta),
            y0 + wez_R * np.sin(theta),
            color="tab:blue", alpha=0.5)

    lead_x, lead_y = _compute_lead_point(s[0], s[1])
    ax.scatter([lead_x], [lead_y], c="cyan", s=40, marker="x")

    for sx, sy, spsi, life, color in shots:
        ax.plot(
            [sx, sx + 600 * np.cos(spsi)],
            [sy, sy + 600 * np.sin(spsi)],
            color=color, alpha=life, lw=1.5
        )

    for fx, fy, life in flashes:
        r = 80 + (1 - life) * 100
        ax.add_artist(
            plt.Circle((fx, fy), r, fill=False,
                       edgecolor="yellow", alpha=life, lw=2)
        )

    ax.text(
        -arena * 0.95, arena * 0.92,
        f"HP0={env_base.hp[0]:.2f}  HP1={env_base.hp[1]:.2f}\n"
        f"Ammo0={env_base.ammo[0]}  Ammo1={env_base.ammo[1]}",
        fontsize=10,
        bbox=dict(facecolor="white", alpha=0.6, edgecolor="none")
    )

    if "winner" in info:
        ax.set_title(
            f"Winner: {'BLUE' if info['winner']==0 else 'RED'}",
            color="tab:blue" if info["winner"] == 0 else "tab:red"
        )
    else:
        ax.set_title("Dogfight Replay")


# ======================================================
# RENDER + EVAL (FIXED)
# ======================================================
def render_eval_episode(
    model,
    make_env_fn,
    save_path,
    norm_obs_fn,          # callable: obs -> normalized obs
    fps=10,
    max_steps=1200,
    deterministic=True,
    slowdown=2,
):
    env = make_env_fn()
    obs, _ = env.reset()

    fig, ax = plt.subplots(figsize=(6, 6))
    writer = PillowWriter(fps=fps)

    shots, flashes = [], []
    trail0, trail1 = [], []
    TRAIL_LEN = 120

    prev_hp = env.base.hp.copy()
    frames_written = 0
    info = {}

    with writer.saving(fig, save_path, dpi=150):
        lstm_state = None
        episode_start = np.array([True], dtype=bool)

        for _ in range(max_steps):

            obs_all = env.base._obs_all()
            a1 = heuristic_policy_continuous(obs_all[1])

            obs_n = norm_obs_fn(obs)
            action, lstm_state = model.predict(
                obs_n,
                state=lstm_state,
                episode_start=episode_start,
                deterministic=deterministic,
            )
            episode_start[:] = False

            s = env.base.s
            x0, y0, _, psi0, _ = s[0]
            x1, y1, _, psi1, _ = s[1]

            a0 = np.asarray(action).ravel()
            if a0.size >= 3 and a0[2] > 0.5 and env.base.ammo[0] > 0:
                shots.append((x0, y0, psi0, 1.0, "tab:blue"))
            if a1[2] > 0.5 and env.base.ammo[1] > 0:
                shots.append((x1, y1, psi1, 1.0, "tab:red"))

            obs, _, done, _, info = env.step(action)

            trail0.append((x0, y0))
            trail1.append((x1, y1))
            if len(trail0) > TRAIL_LEN: trail0.pop(0)
            if len(trail1) > TRAIL_LEN: trail1.pop(0)

            if env.base.hp[1] < prev_hp[1]:
                flashes.append((x1, y1, 1.0))
            if env.base.hp[0] < prev_hp[0]:
                flashes.append((x0, y0, 1.0))
            prev_hp = env.base.hp.copy()

            shots = [(sx, sy, sp, l-0.05, c) for sx, sy, sp, l, c in shots if l-0.05 > 0]
            flashes = [(fx, fy, l-0.06) for fx, fy, l in flashes if l-0.06 > 0]

            for _ in range(max(1, slowdown)):
                _frame(ax, env.base, info, shots, flashes,
                       trail0, trail1,
                       arena=env.base.arena,
                       wez_R=env.base.wez_R,
                       wez_ang=env.base.wez_ang)
                writer.grab_frame()
                frames_written += 1

            if done:
                break

        if frames_written == 0:
            _frame(ax, env.base, info, shots, flashes,
                   trail0, trail1,
                   arena=env.base.arena,
                   wez_R=env.base.wez_R,
                   wez_ang=env.base.wez_ang)
            writer.grab_frame()

    if "winner" not in info:
        if env.base.hp[0] > env.base.hp[1]:
            info["winner"] = 0
        elif env.base.hp[1] > env.base.hp[0]:
            info["winner"] = 1
        else:
            info["winner"] = None

    plt.close(fig)
    return info
