# replay_recorder.py
# Basit dogfight replay kaydı: tek bölüm simüle et, matplotlib ile MP4/GIF üret.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, PillowWriter

def _extract_xy_psi(env_base):
    # env.base.s -> shape (2,5): [x,y,v,psi,bank]
    s = env_base.s
    (x0, y0, _, psi0, _), (x1, y1, _, psi1, _) = s
    return (x0, y0, psi0), (x1, y1, psi1)

def _frame(ax, env_base, info, arena=4000.0, wez_R=800.0, wez_ang=np.deg2rad(25)):
    ax.clear()
    ax.set_aspect('equal', 'box')
    ax.set_xlim(-arena, arena)
    ax.set_ylim(-arena, arena)
    ax.grid(True, alpha=0.2)

    (x0, y0, psi0), (x1, y1, psi1) = _extract_xy_psi(env_base)

    # arena & origin
    ax.add_artist(plt.Circle((0, 0), arena, fill=False, lw=1.0, ls='--', alpha=0.3))

    # uçak üçgeni (kısa bir ok)
    def draw_aircraft(x, y, psi, color):
        L = 100.0  # ok uzunluğu (görsel)
        ax.plot([x, x + L*np.cos(psi)], [y, y + L*np.sin(psi)], color=color, lw=2.0)
        ax.add_artist(plt.Circle((x, y), 25, color=color, fill=True, alpha=0.8))

    draw_aircraft(x0, y0, psi0, 'tab:blue')
    draw_aircraft(x1, y1, psi1, 'tab:red')

    # bearing çizgileri
    ax.plot([x0, x1], [y0, y1], color='gray', lw=1.0, alpha=0.4)

    # WEZ görselleştirme (mavi için)
    # hedefin maviye göre bearing’i
    dx, dy = x1 - x0, y1 - y0
    brg = np.arctan2(dy, dx) - psi0
    brg = (brg + np.pi) % (2*np.pi) - np.pi
    # WEZ sektorünü mavi uçağın burnundan aç
    theta = np.linspace(-wez_ang, wez_ang, 60) + psi0
    wx = x0 + wez_R*np.cos(theta)
    wy = y0 + wez_R*np.sin(theta)
    ax.plot(wx, wy, color='tab:blue', lw=1.0, alpha=0.5)
    ax.text(-arena*0.95, arena*0.92,
            f"HP0={env_base.hp[0]:.2f}  HP1={env_base.hp[1]:.2f}\nAmmo0={env_base.ammo[0]}  Ammo1={env_base.ammo[1]}",
            fontsize=10, bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))

    if "winner" in info:
        w = info["winner"]
        ax.set_title(f"Winner: {'BLUE(0)' if w == 0 else 'RED(1)'}", color=('tab:blue' if w == 0 else 'tab:red'), fontsize=12)
    else:
        ax.set_title("Dogfight replay", fontsize=12)

def _heuristic_continuous(obs0):
    # eğitimde kullandığımız ile uyumlu “temel” heuristic (rakip için)
    R = 2000.0 * float(np.clip(obs0[0], 0.0, 1.0))
    brg = float(np.arctan2(obs0[1], obs0[2]))
    aoff = float(np.arctan2(obs0[3], obs0[4]))
    clos = 200.0 * np.arctanh(np.clip(obs0[5], -0.999, 0.999))
    eps = np.deg2rad(3)
    bank_rate = 0.0
    if brg >  eps: bank_rate = +1.0
    if brg < -eps: bank_rate = -1.0
    throttle = 1.0 if clos < 0 else (0.0 if clos > 120 else 0.5)
    trigger_p = 1.0 if (R < 900 and abs(brg) < np.deg2rad(30) and abs(aoff) < np.deg2rad(25)) else 0.0
    return np.array([bank_rate, throttle, trigger_p], dtype=np.float32)

def render_eval_episode(model, make_env_fn, save_path, vecnorm=None, fps=30, max_steps=1200, deterministic=True):
    """
    model: SB3 (PPO/RecurrentPPO) modeli
    make_env_fn: DogfightSoloEnvPool() döndüren parametresiz fonksiyon
    vecnorm: eğitimdeki VecNormalize (obs_rms kopyalanır)
    save_path: 'runs/.../replay.mp4' veya '.gif'
    """
    # tek süreçli env
    env = make_env_fn()
    obs, _ = env.reset()

    # VecNormalize istatistikleri gerekiyorsa bağla
    if vecnorm is not None:
        # basitçe training=False Norm wrapper’ı olmadan RMS’i uygula:
        def norm_obs(o):
            rms = vecnorm.obs_rms
            if rms is None:
                return o
            return np.clip((o - rms.mean) / np.sqrt(rms.var + 1e-8), -vecnorm.clip_obs, vecnorm.clip_obs)
    else:
        norm_obs = lambda o: o

    fig, ax = plt.subplots(figsize=(6, 6))
    writer = FFMpegWriter(fps=fps) if save_path.lower().endswith(".mp4") else PillowWriter(fps=fps)

    info = {}
    with writer.saving(fig, save_path, dpi=150):
        # recurrent ise state & episode_start yönet
        lstm_state = None
        episode_start = np.array([True], dtype=bool)

        for t in range(max_steps):
            # rakip aksiyonu (heuristic)
            obs_dict = env.base._obs_all()
            a1 = _heuristic_continuous(obs_dict[1])

            # ajan aksiyonu
            o_in = norm_obs(obs)
            # RecurrentPPO mu, PPO mu?
            try:
                action, lstm_state = model.predict(o_in, state=lstm_state, episode_start=episode_start, deterministic=deterministic)
            except TypeError:
                action, _ = model.predict(o_in, deterministic=deterministic)
            episode_start = np.array([False], dtype=bool)

            # env step
            obs_next, rew, done, _, info = env.step(action)
            # çerçeve çiz
            _frame(ax, env.base, info, arena=env.base.arena, wez_R=env.base.wez_R, wez_ang=env.base.wez_ang)
            writer.grab_frame()

            obs = obs_next
            if done:
                # final bir kare daha
                _frame(ax, env.base, info, arena=env.base.arena, wez_R=env.base.wez_R, wez_ang=env.base.wez_ang)
                writer.grab_frame()
                break

    plt.close(fig)
    return info
