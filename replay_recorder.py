# replay_recorder.py
# Gelişmiş dogfight replay:
# - WEZ yayları
# - Mermi rayları
# - Lead-point işareti
# - Hız vektörleri
# - Uçuş trail'i
# - Hit flash

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter   # GIF için

from dogfight_wrappers import heuristic_policy_continuous


def _compute_lead_point(shooter_state, target_state, bullet_speed=600.0):
    """
    Basit 2D sabit-hız lead hesabı (sadece görsel amaçlı).
    shooter_state: (x, y, v, psi, bank)
    target_state:  (x, y, v, psi, bank)
    """
    xs, ys, _, _, _ = shooter_state
    xt, yt, vt, psit, _ = target_state

    # hedef hızı vektörü
    vtx = vt * np.cos(psit)
    vty = vt * np.sin(psit)

    dx = xt - xs
    dy = yt - ys
    R = np.hypot(dx, dy) + 1e-6

    # hedef hızının LOS yönündeki bileşeni
    vrad = (dx * vtx + dy * vty) / R
    # kaba vurma süresi: R / (bullet_speed - vrad)
    denom = bullet_speed - vrad
    if denom <= 1e-3:
        t_hit = R / bullet_speed
    else:
        t_hit = R / denom

    t_hit = np.clip(t_hit, 0.0, 10.0)
    lead_x = xt + vtx * t_hit
    lead_y = yt + vty * t_hit
    return lead_x, lead_y


def _frame(ax, env_base, info, shots, flashes, trail0, trail1,
           arena=4000.0, wez_R=800.0, wez_ang=np.deg2rad(25)):
    ax.clear()
    ax.set_aspect('equal', 'box')
    ax.set_xlim(-arena, arena)
    ax.set_ylim(-arena, arena)
    ax.grid(True, alpha=0.2)

    s = env_base.s   # shape (2,5)
    x0, y0, v0, psi0, _ = s[0]
    x1, y1, v1, psi1, _ = s[1]

    # arena çemberi
    ax.add_artist(plt.Circle((0, 0), arena, fill=False, lw=1.0, ls='--', alpha=0.3))

    # trail çizgileri
    if len(trail0) > 1:
        xs, ys = zip(*trail0)
        ax.plot(xs, ys, color='tab:blue', lw=1.0, alpha=0.5)
    if len(trail1) > 1:
        xs, ys = zip(*trail1)
        ax.plot(xs, ys, color='tab:red', lw=1.0, alpha=0.5)

    # uçak + hız vektörü
    def draw_aircraft(x, y, v, psi, color):
        L = 100.0
        # gövde
        ax.plot([x, x + L * np.cos(psi)], [y, y + L * np.sin(psi)],
                color=color, lw=2.0)
        ax.add_artist(plt.Circle((x, y), 25, color=color, fill=True, alpha=0.8))
        # hız vektörü (speed ile ölçekle)
        v_scale = 1.5   # görsel için sabit
        ax.arrow(x, y,
                 v_scale * v * np.cos(psi),
                 v_scale * v * np.sin(psi),
                 head_width=70.0, head_length=120.0,
                 length_includes_head=True,
                 color=color, alpha=0.6, lw=1.5)

    draw_aircraft(x0, y0, v0, psi0, 'tab:blue')
    draw_aircraft(x1, y1, v1, psi1, 'tab:red')

    # bearing çizgisi (LOS)
    ax.plot([x0, x1], [y0, y1], color='gray', lw=1.0, alpha=0.4)

    # mavi WEZ sektörü
    theta = np.linspace(-wez_ang, wez_ang, 60) + psi0
    wx = x0 + wez_R * np.cos(theta)
    wy = y0 + wez_R * np.sin(theta)
    ax.plot(wx, wy, color='tab:blue', lw=1.0, alpha=0.5)

    # lead-point (mavi mermisi için, hedef kırmızı)
    lead_x, lead_y = _compute_lead_point(s[0], s[1])
    ax.scatter([lead_x], [lead_y], color='cyan', s=40, marker='x', alpha=0.9)

    # mermi rayları (shots listesi)
    # her shot: (x, y, psi, life, color)
    Lb = 600.0
    for (sx, sy, spsi, life, color) in shots:
        alpha = max(0.0, min(1.0, life))
        ax.plot(
            [sx, sx + Lb * np.cos(spsi)],
            [sy, sy + Lb * np.sin(spsi)],
            color=color,
            lw=1.5,
            alpha=alpha,
        )

    # hit flash'ler (flashes listesi)
    # her flash: (x, y, life)
    for (fx, fy, life) in flashes:
        R0 = 80.0
        R1 = 180.0
        r = R0 + (1.0 - life) * (R1 - R0)
        alpha = life
        circ = plt.Circle((fx, fy), r, fill=False, lw=2.0,
                          edgecolor='yellow', alpha=alpha)
        ax.add_artist(circ)

    # üst bilgi yazıları
    ax.text(-arena * 0.95, arena * 0.92,
            f"HP0={env_base.hp[0]:.2f}  HP1={env_base.hp[1]:.2f}\n"
            f"Ammo0={env_base.ammo[0]}  Ammo1={env_base.ammo[1]}",
            fontsize=10,
            bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))

    if "winner" in info:
        w = info["winner"]
        ax.set_title(f"Winner: {'BLUE(0)' if w == 0 else 'RED(1)'}",
                     color=('tab:blue' if w == 0 else 'tab:red'),
                     fontsize=12)
    else:
        ax.set_title("Dogfight replay", fontsize=12)


def render_eval_episode(model, make_env_fn, save_path, vecnorm=None,
                        fps=10, max_steps=1200, deterministic=True,
                        slowdown=2):
    """
    model       : SB3 (RecurrentPPO) modeli
    make_env_fn : DogfightSoloEnvPool() döndüren fonksiyon
    save_path   : .gif dosya yolu
    vecnorm     : eğitimde kullanılan VecNormalize (obs_rms için)
    fps         : GIF kare hızı (env dt=0.1 ise 10 fps ~ gerçek zaman)
    slowdown    : her kareyi kaç kez yazalım (2 -> 2x yavaş)
    """
    env = make_env_fn()
    obs, _ = env.reset()

    # obs normalizasyon fonksiyonu
    if vecnorm is not None:
        def norm_obs(o):
            rms = vecnorm.obs_rms
            if rms is None:
                return o
            return np.clip(
                (o - rms.mean) / np.sqrt(rms.var + 1e-8),
                -vecnorm.clip_obs,
                vecnorm.clip_obs,
            )
    else:
        norm_obs = lambda o: o

    fig, ax = plt.subplots(figsize=(6, 6))
    writer = PillowWriter(fps=fps)

    info = {}
    shots = []      # (x, y, psi, life, color)
    flashes = []    # (x, y, life)
    shot_decay = 0.05    # her framede life -= 0.05
    flash_decay = 0.06

    trail0 = []
    trail1 = []
    TRAIL_LEN = 120   # son 120 nokta

    prev_hp = env.base.hp.copy()

    with writer.saving(fig, save_path, dpi=150):
        lstm_state = None
        episode_start = np.array([True], dtype=bool)

        for t in range(max_steps):
            # env'den obs'leri çek
            obs_dict = env.base._obs_all()

            # rakip aksiyonu (heuristic)
            a1 = heuristic_policy_continuous(obs_dict[1])

            # ajan aksiyonu
            o_in = norm_obs(obs)
            try:
                action, lstm_state = model.predict(
                    o_in,
                    state=lstm_state,
                    episode_start=episode_start,
                    deterministic=deterministic,
                )
            except TypeError:
                action, _ = model.predict(o_in, deterministic=deterministic)
            episode_start[:] = False

            # adım öncesi state (mermi başlangıç noktası)
            s = env.base.s
            x0, y0, _, psi0, _ = s[0]
            x1, y1, _, psi1, _ = s[1]

            # tetik durumları
            a0 = np.asarray(action, dtype=float).ravel()
            trig0 = float(a0[2]) if a0.size >= 3 else 0.0
            trig1 = float(a1[2]) if a1.size >= 3 else 0.0

            if trig0 > 0.5 and env.base.ammo[0] > 0:
                shots.append((x0, y0, psi0, 1.0, 'tab:blue'))
            if trig1 > 0.5 and env.base.ammo[1] > 0:
                shots.append((x1, y1, psi1, 1.0, 'tab:red'))

            # env adımı
            obs_next, rew, done, _, info = env.step(action)

            # trail güncelle
            trail0.append((x0, y0))
            trail1.append((x1, y1))
            if len(trail0) > TRAIL_LEN:
                trail0.pop(0)
            if len(trail1) > TRAIL_LEN:
                trail1.pop(0)

            # hp düşmüş mü? hit flash ekle
            hp = env.base.hp
            # mavi kırmızıya vurmuşsa hp1 azalır
            if hp[1] < prev_hp[1] - 1e-6:
                flashes.append((x1, y1, 1.0))
            # kırmızı maviyi vurmuşsa hp0 azalır
            if hp[0] < prev_hp[0] - 1e-6:
                flashes.append((x0, y0, 1.0))
            prev_hp = hp.copy()

            # shot & flash life'ları azalt
            new_shots = []
            for (sx, sy, spsi, life, color) in shots:
                life_new = life - shot_decay
                if life_new > 0:
                    new_shots.append((sx, sy, spsi, life_new, color))
            shots = new_shots

            new_flashes = []
            for (fx, fy, life) in flashes:
                life_new = life - flash_decay
                if life_new > 0:
                    new_flashes.append((fx, fy, life_new))
            flashes = new_flashes

            # kare çiz ve slowdown için tekrar et
            for _ in range(max(1, slowdown)):
                _frame(ax, env.base, info, shots, flashes,
                       trail0, trail1,
                       arena=env.base.arena,
                       wez_R=env.base.wez_R,
                       wez_ang=env.base.wez_ang)
                writer.grab_frame()

            obs = obs_next
            if done:
                for _ in range(max(1, slowdown)):
                    _frame(ax, env.base, info, shots, flashes,
                           trail0, trail1,
                           arena=env.base.arena,
                           wez_R=env.base.wez_R,
                           wez_ang=env.base.wez_ang)
                    writer.grab_frame()
                break

    plt.close(fig)
    return info
