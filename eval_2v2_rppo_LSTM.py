import os
import json
import numpy as np

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from scenarios.scenario_2v2.dogfight_env_2v2_3dof import Dogfight2v2_3DOF
from scenarios.scenario_2v2.dogfight_wrappers_2v2_3dof import Dogfight2v2SB3Wrapper
from tools.plot_eval_summary import plot_eval_summary


# ======================================================
# CONFIG
# ======================================================
RUN_NAME = "rppo_2v2_final_reward_v3"
RUN_DIR = os.path.join("runs/runs_2v2", RUN_NAME)

EVAL_DIR = os.path.join(RUN_DIR, "eval")
REPLAY_DIR = os.path.join(EVAL_DIR, "replays")
os.makedirs(REPLAY_DIR, exist_ok=True)

N_EVAL_EPISODES = 50
SEED_START = 1000


# ======================================================
# LOAD MODEL
# ======================================================
model = RecurrentPPO.load(os.path.join(RUN_DIR, "final_model"))


# ======================================================
# BUILD ENV ONCE
# ======================================================
def make_eval_env(seed):
    base_env = Dogfight2v2_3DOF(seed=seed)
    return Dogfight2v2SB3Wrapper(base_env)


all_logs = []

for ep in range(N_EVAL_EPISODES):
    seed = SEED_START + ep

    vec_env = DummyVecEnv([lambda s=seed: make_eval_env(s)])
    vec_env = VecNormalize.load(
        os.path.join(RUN_DIR, "vecnormalize.pkl"),
        vec_env
    )

    vec_env.training = False
    vec_env.norm_reward = False

    # 🔥 DOĞRU ENV SEVİYESİ
    base_env = vec_env.envs[0].env
    base_env.enable_logging = True
    base_env.log_dir = REPLAY_DIR
    base_env.episode_tag = f"{ep:05d}"

    # --------------------------------------------------
    # RESET SADECE BİR KEZ
    # --------------------------------------------------
    obs = vec_env.reset()
    done = False
    step_count = 0
    info = {}

    lstm_state = None
    episode_start = np.ones((1,), dtype=bool)

    print(f"\n=== EVAL EPISODE {ep} | SEED {seed} ===")

    while not done:
        action, lstm_state = model.predict(
            obs,
            state=lstm_state,
            episode_start=episode_start,
            deterministic=True,
        )

        obs, _, done, info = vec_env.step(action)
        episode_start = done
        step_count += 1

    info0 = info[0] if isinstance(info, list) else info

    log = {
        "episode": ep,
        "seed": seed,
        "winner": info0.get("winner", "unknown"),
        "termination": info0.get("termination", "unknown"),
        "min_dist": info0.get("min_dist"),
        "avg_dist": info0.get("avg_dist"),
        "episode_len": step_count,
    }

    all_logs.append(log)
    print("EVAL:", log)


# ======================================================
# SAVE SUMMARY
# ======================================================
summary_path = os.path.join(EVAL_DIR, "eval_summary.json")
with open(summary_path, "w") as f:
    json.dump(all_logs, f, indent=2)


# ======================================================
# WIN-RATE PLOT
# ======================================================
plot_eval_summary(summary_path, EVAL_DIR)

print("\nEVAL FINISHED")
