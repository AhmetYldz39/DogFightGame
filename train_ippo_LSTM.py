# train_ippo_LSTM.py
# RecurrentPPO (SB3-Contrib) + Separate LSTM (actor & critic) + Continuous Action
# Başlangıçta yalnızca HEURISTIC rakip (continuous). Snapshot havuzu sonra açacağız.

import os
import time
import numpy as np
import torch.nn as nn

from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize, VecMonitor
from stable_baselines3.common.logger import configure
from sb3_contrib import RecurrentPPO
from sb3_contrib.ppo_recurrent import MlpLstmPolicy

from train_ippo_pool import DogfightSoloEnvPool  # wrapper'ın continuous & sin/cos sürümü

# -------------------------
# KONFİG
# -------------------------
RUN_NAME_PREFIX = "rppo_sepLSTM_heur"
N_ENVS = 8
EVAL_EPISODES = 200
TOTAL_STEPS = 900_000
# Entropy annealing: yüksek başlat, kademeli indir
ENTROPY_SCHEDULE = [(0, 0.03), (300_000, 0.02), (600_000, 0.012), (850_000, 0.008)]

# -------------------------
# ENV FABRİKALARI
# -------------------------
def make_train_env(seed_base=10_000):
    """SubprocVecEnv için environment thunk listesi döndürür."""
    thunks = []
    for i in range(N_ENVS):
        def _thunk(rank=i):
            def _f():
                # Sadece heuristic ile başla (continuous)
                return DogfightSoloEnvPool(
                    seed=seed_base + rank,
                    opponent_pool=['heuristic'],
                    pool_probs=[1.0],
                )
            return _f
        thunks.append(_thunk())
    return thunks

def make_eval_env(seed_base=99_000, n=1):
    thunks = []
    for i in range(n):
        def _thunk(rank=i):
            def _f():
                return DogfightSoloEnvPool(
                    seed=seed_base + rank,
                    opponent_pool=['heuristic'],
                    pool_probs=[1.0],
                )
            return _f
        thunks.append(_thunk())
    return thunks

# -------------------------
# LSTM UYUMLU EVAL (win-rate)
# -------------------------
def evaluate_win_rate(model, train_vecnorm, episodes=EVAL_EPISODES, seed=4242):
    def _make():
        return DogfightSoloEnvPool(seed=seed, opponent_pool=['heuristic'], pool_probs=[1.0])

    eval_env = DummyVecEnv([_make])
    eval_env = VecMonitor(eval_env, filename=None)
    eval_norm = VecNormalize(eval_env, training=False, norm_obs=True, norm_reward=False)
    # eğitim istatistiklerini eval'e kopyala
    eval_norm.obs_rms = train_vecnorm.obs_rms

    wins = 0
    for _ in range(episodes):
        obs = eval_norm.reset()
        lstm_state = None
        episode_start = np.array([True], dtype=bool)
        done = False
        last_info = {}
        while not done:
            action, lstm_state = model.predict(
                obs, state=lstm_state, episode_start=episode_start, deterministic=True
            )
            obs, reward, done_vec, infos = eval_norm.step(action)
            episode_start = done_vec
            done = bool(done_vec[0])
            if done:
                last_info = infos[0]
        if last_info.get("winner", None) == 0:
            wins += 1

    eval_norm.close()
    return wins / episodes

# -------------------------
# MAIN
# -------------------------
if __name__ == "__main__":
    run_name = time.strftime(f"{RUN_NAME_PREFIX}_%Y%m%d_%H%M%S")
    log_dir = os.path.join("runs", run_name)
    os.makedirs(log_dir, exist_ok=True)

    # --- Train env ---
    train_env = SubprocVecEnv(make_train_env())
    train_env = VecMonitor(train_env, filename=None)
    train_vecnorm = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=5.0)

    # --- Logger ---
    logger = configure(log_dir, ["stdout", "tensorboard"])

    # --- RecurrentPPO: Separate LSTM (actor & critic) ---
    policy_kwargs = dict(
        lstm_hidden_size=128,
        n_lstm_layers=1,
        shared_lstm=False,        # separate policy
        enable_critic_lstm=True,  # critic'in kendi LSTM'i var
        net_arch=dict(pi=[128], vf=[128]),
        activation_fn=nn.Tanh,
    )

    model = RecurrentPPO(
        policy=MlpLstmPolicy,
        env=train_vecnorm,
        policy_kwargs=policy_kwargs,
        n_steps=256,                  # LSTM için kısa rollout
        batch_size=N_ENVS * 64,       # 512
        n_epochs=4,
        learning_rate=2e-4,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.25,
        ent_coef=ENTROPY_SCHEDULE[0][1],   # 0.03 ile başla
        vf_coef=0.5,
        verbose=1,
        tensorboard_log=log_dir,
        seed=42,
    )
    model.set_logger(logger)

    # --- Eğitim + Entropy Anneal ---
    step_cursor = 0
    next_schedule_idx = 1  # 0.03 uygulandı; sıradaki 0.02
    while step_cursor < TOTAL_STEPS:
        chunk = min(100_000, TOTAL_STEPS - step_cursor)
        model.learn(total_timesteps=chunk, reset_num_timesteps=False)
        step_cursor += chunk

        # Entropy schedule: basit eşik kontrolü
        while next_schedule_idx < len(ENTROPY_SCHEDULE) and step_cursor >= ENTROPY_SCHEDULE[next_schedule_idx][0]:
            model.ent_coef = ENTROPY_SCHEDULE[next_schedule_idx][1]
            print(f"[ANNEAL] ent_coef -> {model.ent_coef} @ {step_cursor}")
            next_schedule_idx += 1

        # Ara değerlendirme
        wr = evaluate_win_rate(model, train_vecnorm, episodes=100)
        print(f"[STEP {step_cursor}] WR vs HEUR(100): {wr:.2%}")

    # --- Final kayıt ---
    final_model_path = os.path.join(log_dir, "final_model.zip")
    model.save(final_model_path)
    train_vecnorm.save(os.path.join(log_dir, "vecnormalize.pkl"))
    print(f"[INFO] Final model saved: {final_model_path}")

    # --- Geniş değerlendirme ---
    wr = evaluate_win_rate(model, train_vecnorm, episodes=EVAL_EPISODES)
    print(f"[EVAL] Win-rate vs HEURISTIC ({EVAL_EPISODES} ep): {wr:.2%}")
    print(f"[TENSORBOARD] tensorboard --logdir {log_dir}")
