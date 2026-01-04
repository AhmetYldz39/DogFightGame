# =========================
# train_1v1_ippo_LSTM.py
# =========================
import os
import time
import numpy as np
import torch.nn as nn

from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, VecMonitor
from stable_baselines3.common.logger import configure
from sb3_contrib import RecurrentPPO
from sb3_contrib.ppo_recurrent import MlpLstmPolicy

from scenarios.scenario_1v1.dogfight_wrappers import DogfightSoloEnvPool
from tools.replay_recorder import render_eval_episode


# -------------------------------------------------------
# True = fine-tune present model, False = train new model
LOAD_PRESENT_MODEL = True
# -------------------------------------------------------

# -------------------------
# config
# -------------------------
RUN_NAME_PREFIX = "rppo_sepLSTM_frozenExpert"
N_ENVS = 8
EVAL_EPISODES = 100

# total steps
TOTAL_STEPS = 900_000
FINE_TUNE_STEPS = 400_000

# entropy anneal
ENTROPY_SCHEDULE = [(0, 0.02), (250_000, 0.015), (500_000, 0.01), (800_000, 0.008)]

# -------------------
# EXPERT OPPONENT (frozen)
# -------------------
EXPERT_RUN_DIR = r"runs\runs_1v1\rppo_sepLSTM_heur_20251206_234439"
EXPERT_MODEL_PATH = os.path.join(EXPERT_RUN_DIR, "fine_tune_model_v3.zip")
EXPERT_VEC_PATH = os.path.join(EXPERT_RUN_DIR, "vecnormalize.pkl")

# opponent curriculum:
OPPONENT_POOL = [
    "heuristic",
    ("expert", EXPERT_MODEL_PATH, EXPERT_VEC_PATH),
]
OPPONENT_PROBS = [0.8, 0.2]  # başlangıç: %80 heuristic, %20 expert


# -------------------------
# ENVIRONMENTS
# -------------------------
def make_train_env(seed_base=10_000):
    thunks = []
    for i in range(N_ENVS):
        def _thunk(rank=i):
            def _f():
                return DogfightSoloEnvPool(
                    seed=seed_base + rank,
                    opponent_pool=OPPONENT_POOL,
                    pool_probs=OPPONENT_PROBS,
                    timeout_as_loss=False,
                    expert_device="cpu",
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
                    opponent_pool=OPPONENT_POOL,
                    pool_probs=OPPONENT_PROBS,
                    timeout_as_loss=True,
                    expert_device="cpu",
                )
            return _f
        thunks.append(_thunk())
    return thunks


# -------------------------
# LSTM UYUMLU EVAL (win-rate)
# -------------------------
def evaluate_win_rate(model, vec_env_norm: VecNormalize, episodes=100, max_steps_per_episode=1500):
    env = vec_env_norm
    n_envs = env.num_envs

    # freeze VecNormalize statistics
    old_training_flag = getattr(env, "training", None)
    if hasattr(env, "training"):
        env.training = False

    obs = env.reset()
    lstm_states = None
    episode_starts = np.ones(n_envs, dtype=bool)

    wins = 0
    total_eps = 0
    global_steps = 0
    max_global_steps = episodes * max_steps_per_episode

    while total_eps < episodes and global_steps < max_global_steps:
        actions, lstm_states = model.predict(
            obs,
            state=lstm_states,
            episode_start=episode_starts,
            deterministic=True,
        )
        obs, rewards, dones, infos = env.step(actions)
        episode_starts = dones
        global_steps += n_envs

        for i, done in enumerate(dones):
            if done:
                winner = infos[i].get("winner", -1)
                if winner == 0:
                    wins += 1
                total_eps += 1
                if total_eps >= episodes:
                    break

    if old_training_flag is not None:
        env.training = old_training_flag

    if total_eps == 0:
        return 0.0
    return wins / total_eps


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

    # --- Policy kwargs (separate LSTM) ---
    policy_kwargs = dict(
        lstm_hidden_size=128,
        n_lstm_layers=1,
        shared_lstm=False,
        enable_critic_lstm=True,
        net_arch=dict(pi=[128], vf=[128]),
        activation_fn=nn.Tanh,
    )

    if LOAD_PRESENT_MODEL:
        # Fine-tune: kendi mevcut challenger modelini buradan load edebilirsin
        # Eğer bu scripti ilk kez expert'e karşı koşacaksan:
        # - LOAD_PRESENT_MODEL=False ile yeni model eğit,
        # - ya da mevcut iyi modelini buraya path ver.
        LOAD_FROM = EXPERT_MODEL_PATH
        print(f"[FINE-TUNE] Loading model from: {LOAD_FROM}")
        model = RecurrentPPO.load(LOAD_FROM, env=train_vecnorm)

        # Fine-tune hyperparam
        model.learning_rate = 1e-4
        model.ent_coef = 0.01

        model.set_logger(logger)
        model.learn(total_timesteps=FINE_TUNE_STEPS, reset_num_timesteps=False)
        save_name = "challenger_vs_expert.zip"
    else:
        model = RecurrentPPO(
            policy=MlpLstmPolicy,
            env=train_vecnorm,
            policy_kwargs=policy_kwargs,
            n_steps=256,
            batch_size=N_ENVS * 64,
            n_epochs=4,
            learning_rate=2e-4,
            gamma=0.995,
            gae_lambda=0.95,
            clip_range=0.25,
            ent_coef=ENTROPY_SCHEDULE[0][1],
            vf_coef=0.5,
            verbose=1,
            tensorboard_log=log_dir,
            seed=42,
        )
        model.set_logger(logger)

        step_cursor = 0
        next_schedule_idx = 1
        while step_cursor < TOTAL_STEPS:
            chunk = min(100_000, TOTAL_STEPS - step_cursor)
            model.learn(total_timesteps=chunk, reset_num_timesteps=False)
            step_cursor += chunk

            # anneal
            while next_schedule_idx < len(ENTROPY_SCHEDULE) and step_cursor >= ENTROPY_SCHEDULE[next_schedule_idx][0]:
                model.ent_coef = ENTROPY_SCHEDULE[next_schedule_idx][1]
                print(f"[ANNEAL] ent_coef -> {model.ent_coef} @ {step_cursor}")
                next_schedule_idx += 1

            wr = evaluate_win_rate(model, train_vecnorm, episodes=50)
            print(f"[STEP {step_cursor}] WR pool(50): {wr:.2%}")

        save_name = "challenger_new_vs_expert.zip"

    # --- Save ---
    final_model_path = os.path.join(log_dir, save_name)
    model.save(final_model_path)
    train_vecnorm.save(os.path.join(log_dir, "vecnormalize.pkl"))
    print(f"[INFO] Saved: {final_model_path}")

    # --- EVAL ---
    wr = evaluate_win_rate(model, train_vecnorm, episodes=EVAL_EPISODES)
    print(f"[EVAL] Win-rate pool ({EVAL_EPISODES} ep): {wr:.2%}")
    print(f"[TENSORBOARD] tensorboard --logdir {log_dir}")

    # --- REPLAY ---
    replay_path = os.path.join(log_dir, "replay_pool.gif")

    def make_env_fn():
        # Replay'de de aynı pool'u kullan (istersen burada probs'u expert ağırlıklı yapabilirsin)
        return DogfightSoloEnvPool(
            seed=123,
            opponent_pool=OPPONENT_POOL,
            pool_probs=OPPONENT_PROBS,
            timeout_as_loss=True,   # replay'de net bitiş
            expert_device="cpu",
        )

    info_last = render_eval_episode(
        model=model,
        make_env_fn=make_env_fn,
        save_path=replay_path,
        vecnorm=train_vecnorm,     # replay recorder bunu kullanıyorsa (senin versiyona göre)
        fps=30,
        max_steps=1200,
        deterministic=True,
    )
    print("[REPLAY] saved:", replay_path, "| info:", info_last)
