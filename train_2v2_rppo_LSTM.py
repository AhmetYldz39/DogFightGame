import os
import json

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback

from scenarios.scenario_2v2.dogfight_env_2v2_3dof import Dogfight2v2_3DOF
from scenarios.scenario_2v2.dogfight_wrappers_2v2_3dof import Dogfight2v2SB3Wrapper


# ======================================================
# CONFIG
# ======================================================
RUN_NAME = "rppo_2v2_final_reward_v2"
RUN_DIR = os.path.join("runs/runs_2v2", RUN_NAME)
os.makedirs(RUN_DIR, exist_ok=True)

SEED = 42
N_ENVS = 8
TOTAL_STEPS = [150_000, 100_000, 50_000]  # -> for each phase
ENT_COEF = 0.02

# Fine-tune opsiyonu
FINE_TUNE = False
BASE_RUN = "rppo_2v2_prev_run"  # FINE_TUNE=True ise kullanılır


# ======================================================
# ENV FACTORY (TRAIN)
# ======================================================
def make_env(rank):
    def _init():
        env = Dogfight2v2_3DOF(seed=SEED + rank)
        env.enable_logging = False            # TRAIN'DE REPLAY YOK
        env = Dogfight2v2SB3Wrapper(env)
        return env
    return _init


vec_env = DummyVecEnv([make_env(i) for i in range(N_ENVS)])


# ------------------------------------------------------
# VecNormalize
# ------------------------------------------------------
if FINE_TUNE:
    vec_env = VecNormalize.load(
        os.path.join("runs", BASE_RUN, "vecnormalize.pkl"),
        vec_env
    )
else:
    vec_env = VecNormalize(
        vec_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0
    )


# ======================================================
# MODEL
# ======================================================
if FINE_TUNE:
    model = RecurrentPPO.load(
        os.path.join("runs", BASE_RUN, "final_model.zip"),
        env=vec_env
    )
else:
    model = RecurrentPPO(
        "MlpLstmPolicy",
        vec_env,
        n_steps=256,
        batch_size=512,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=ENT_COEF,
        verbose=1,
        tensorboard_log=RUN_DIR,
        seed=SEED,
    )


# ======================================================
# CALLBACK
# ======================================================
checkpoint_cb = CheckpointCallback(
    save_freq=200_000 // N_ENVS,
    save_path=RUN_DIR,
    name_prefix="ckpt",
    save_vecnormalize=True,
)


# =========================
# TRAIN – PHASED
# =========================

# Phase 1 – Exploration
model.learn(
    total_timesteps=TOTAL_STEPS[0],
    callback=checkpoint_cb,
    reset_num_timesteps=not FINE_TUNE,
    progress_bar=True,
)
print("Entropy set to 0.02 → 0.01")

# Phase 2 – Transition
model.ent_coef = 0.01
model.learn(
    total_timesteps=TOTAL_STEPS[1],
    callback=checkpoint_cb,
    reset_num_timesteps=False,   # ÇOK ÖNEMLİ
    progress_bar=True,
)
print("Entropy set to 0.01 → 0.005")

# Phase 3 – Exploitation
model.ent_coef = 0.005
model.learn(
    total_timesteps=TOTAL_STEPS[2],
    callback=checkpoint_cb,
    reset_num_timesteps=False,
    progress_bar=True,
)


model.save(os.path.join(RUN_DIR, "final_model.zip"))
vec_env.save(os.path.join(RUN_DIR, "vecnormalize.pkl"))


# ======================================================
# RUN METADATA (çok önemli)
# ======================================================
run_info = {
    "run_name": RUN_NAME,
    "reward_shaping": "distance + WEZ + fire + time penalty",
    "fine_tune": FINE_TUNE,
    "total_steps": TOTAL_STEPS,
    "n_envs": N_ENVS,
}

with open(os.path.join(RUN_DIR, "run_info.json"), "w") as f:
    json.dump(run_info, f, indent=2)

print("TRAIN FINISHED:", RUN_NAME)
