import numpy as np
import glob
import os
import random


def select_best_episode(replay_dir):
    best_win_score = -1e9
    best_win_file = None

    best_any_score = -1e9
    best_any_file = None

    files = glob.glob(os.path.join(replay_dir, "*.npz"))
    if len(files) == 0:
        return None

    for f in files:
        data = np.load(f, allow_pickle=True)

        rewards = data["reward"]   # (T,) object, each = {0: r0, 1: r1}
        hp_final = data["hp"][-1]

        total_reward = 0.0
        for r in rewards:
            total_reward += r.get(0, 0.0) + r.get(1, 0.0)

        teamA_alive = (hp_final[0] > 0) or (hp_final[1] > 0)
        teamB_dead = (hp_final[2] <= 0) and (hp_final[3] <= 0)

        # -------------------------
        # WIN episode (öncelik)
        # -------------------------
        if teamA_alive and teamB_dead:
            if total_reward > best_win_score:
                best_win_score = total_reward
                best_win_file = f

        # -------------------------
        # ANY episode (fallback)
        # -------------------------
        if total_reward > best_any_score:
            best_any_score = total_reward
            best_any_file = f

    # Önce win varsa onu döndür
    if best_win_file is not None:
        print("✅ Winning episode selected.")
        return best_win_file

    # Yoksa fallback
    print("⚠️ No win found. Fallback: best reward episode selected.")
    return best_any_file
