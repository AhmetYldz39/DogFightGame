import os
import numpy as np
import glob

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback

from scenarios.scenario_2v2.dogfight_env_2v2_3dof import Dogfight2v2_3DOF
from scenarios.scenario_2v2.dogfight_wrappers_2v2_3dof import Dogfight2v2SB3Wrapper

from tools.replay_animate_2v2_3dof import animate_replay
from tools.select_best_episode import select_best_episode


RUN_NAME = "rppo_2v2_3dof_heuristic_v2"
LOG_DIR = os.path.join("runs", RUN_NAME)
os.makedirs(LOG_DIR, exist_ok=True)

SEED = 42
N_ENVS = 8
TOTAL_STEPS = 1_000_000


def make_env(seed_offset):
    def _init():
        env = Dogfight2v2_3DOF(seed=SEED + seed_offset)
        env.enable_logging = True
        env.log_dir = os.path.join(LOG_DIR, "replays", f"env_{seed_offset}")
        env = Dogfight2v2SB3Wrapper(env)
        return env
    return _init


vec_env = DummyVecEnv([make_env(i) for i in range(N_ENVS)])
vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

model = RecurrentPPO(
    "MlpLstmPolicy",
    vec_env,
    n_steps=256,
    batch_size=512,
    learning_rate=3e-4,
    gamma=0.99,
    gae_lambda=0.95,
    ent_coef=0.01,
    verbose=1,
    tensorboard_log=LOG_DIR,
    seed=SEED,
)

checkpoint_cb = CheckpointCallback(
    save_freq=200_000 // N_ENVS,
    save_path=LOG_DIR,
    name_prefix="rppo_2v2",
    save_vecnormalize=True,
)

model.learn(
    total_timesteps=TOTAL_STEPS,
    callback=checkpoint_cb,
    progress_bar=True,
)

model.save(os.path.join(LOG_DIR, "final_model.zip"))
vec_env.save(os.path.join(LOG_DIR, "vecnormalize.pkl"))

# ---------------- EVAL (DETERMINISTIC) ----------------
print("Running deterministic eval...")

eval_env = Dogfight2v2_3DOF(seed=999)
eval_env.enable_logging = True
eval_env.log_dir = os.path.join(LOG_DIR, "eval_replays")
eval_env = Dogfight2v2SB3Wrapper(eval_env)

eval_vec = VecNormalize.load(
    os.path.join(LOG_DIR, "vecnormalize.pkl"),
    eval_env
)
eval_vec.training = False
eval_vec.norm_reward = False

obs, _ = eval_vec.reset()
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
    obs, _, done, _, info = eval_vec.step(action)
    episode_start = np.array([done], dtype=bool)

print("Eval finished:", info)

# --------------- BEST EPISODE & VIDEO -----------------
best = select_best_episode(os.path.join(LOG_DIR, "eval_replays"))
if best:
    animate_replay(
        best,
        save_path=best.replace(".npz", "_BEST.mp4"),
        fps=30,
    )
else:
    print("No winning eval episode found.")
