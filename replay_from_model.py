# replay_from_model.py

import os
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

from train_ippo_pool import DogfightSoloEnvPool
from replay_recorder import render_eval_episode

# 1) Hangi run'ı kullanacağımız
RUN_DIR = r"runs\rppo_sepLSTM_heur_20251109_232402"
MODEL_NAME = "final_model"          # .zip yazma, SB3 kendisi ekliyor

model_path = os.path.join(RUN_DIR, MODEL_NAME)
vec_path = os.path.join(RUN_DIR, "vecnormalize.pkl")

# 2) Modeli yükle
model = RecurrentPPO.load(model_path)

# 3) Env factory (hem dummy, hem replay için)
def make_env():
    return DogfightSoloEnvPool(
        seed=123,
        opponent_pool=['heuristic'],
        pool_probs=[1.0],
    )

# 4) VecNormalize istatistiklerini yükle
dummy_env = DummyVecEnv([make_env])          # sadece load için kullanılacak
train_vecnorm = VecNormalize.load(vec_path, dummy_env)
train_vecnorm.training = False
train_vecnorm.norm_reward = False

# 5) Replay kaydet
save_path = os.path.join(RUN_DIR, "replay_final_model.gif")

info_last = render_eval_episode(
    model=model,
    make_env_fn=make_env,
    save_path=save_path,
    vecnorm=train_vecnorm,     # içinden sadece obs_rms & clip_obs okuyacağız
    fps=10,
    max_steps=1200,
    deterministic=True,
    slowdown=2
)

print("[REPLAY] saved:", save_path)
print("[INFO]", info_last)
