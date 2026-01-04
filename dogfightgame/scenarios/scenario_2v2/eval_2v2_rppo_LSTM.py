import os
import sys
import numpy as np

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from scenarios.scenario_2v2.dogfight_env_2v2_3dof import Dogfight2v2_3DOF
from scenarios.scenario_2v2.dogfight_wrappers_2v2_3dof import Dogfight2v2SB3Wrapper
from tools.select_best_episode import select_best_episode
from tools.replay_animate_2v2_3dof import animate_replay


# =====================================================
# Resolve PROJECT ROOT
# =====================================================
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# =====================================================
# CONFIG
# =====================================================
RUN_NAME = "rppo_2v2_3dof_heuristic_v0"
RUN_DIR = os.path.join(PROJECT_ROOT, "runs/runs_2v2", RUN_NAME)

MODEL_PATH = os.path.join(RUN_DIR, "rppo_2v2_1000000_steps")
VECNORM_PATH = os.path.join(RUN_DIR, "vecnormalize.pkl")

EVAL_EPISODES = 5
SEED_START = 1000


# =====================================================
# Load trained model
# =====================================================
print("Loading model:", MODEL_PATH)
model = RecurrentPPO.load(MODEL_PATH)


# =====================================================
# Eval env factory (DummyVecEnv için)
# =====================================================
def make_eval_env(seed, log_dir):
    def _init():
        env = Dogfight2v2_3DOF(seed=seed)
        env.enable_logging = True
        env.log_dir = log_dir
        env = Dogfight2v2SB3Wrapper(env)
        return env
    return _init


# =====================================================
# Run evaluation episodes
# =====================================================
eval_replay_dir = os.path.join(RUN_DIR, "eval_replays")
os.makedirs(eval_replay_dir, exist_ok=True)

for ep in range(EVAL_EPISODES):
    seed = SEED_START + ep
    print(f"\nRunning eval episode {ep+1}/{EVAL_EPISODES} | seed={seed}")

    # --- VecEnv ---
    vec_env = DummyVecEnv([make_eval_env(seed, eval_replay_dir)])

    # --- Load VecNormalize stats ---
    vec_env = VecNormalize.load(VECNORM_PATH, vec_env)
    vec_env.training = False
    vec_env.norm_reward = False

    obs = vec_env.reset()
    done = False

    lstm_state = None
    episode_start = np.ones((1,), dtype=bool)

    while not done:
        action, lstm_state = model.predict(
            obs,
            state=lstm_state,
            episode_start=episode_start,
            deterministic=True,
        )

        obs, _, done, _ = vec_env.step(action)
        episode_start = done

    print("Eval episode finished.")


# =====================================================
# Select best episode
# =====================================================
best_episode = select_best_episode(eval_replay_dir)

if best_episode is None:
    print("\n❌ No winning eval episode found.")
else:
    print("\n🏆 Best episode:", best_episode)

    # =================================================
    # Animate best episode
    # =================================================
    animate_replay(
        best_episode,
        save_path=best_episode.replace(".npz", "_BEST.gif"),
        fps=30,
    )

    print("🎬 Best replay video saved.")
