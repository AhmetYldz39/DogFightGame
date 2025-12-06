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

from dogfight_wrappers import DogfightSoloEnvPool
from replay_recorder import render_eval_episode

# -------------------------------------------------------
# True = enhance a present model, False = train new model
LOAD_PRESENT_MODEL = True
# -------------------------------------------------------

# -------------------------
# config
# -------------------------
RUN_NAME_PREFIX = "rppo_sepLSTM_heur"
N_ENVS = 8
EVAL_EPISODES = 200
TOTAL_STEPS = 900_000
# Entropy annealing: yüksek başlat, kademeli indir
ENTROPY_SCHEDULE = [(0, 0.03), (300_000, 0.02), (600_000, 0.012), (850_000, 0.008)]
SAVE_NAME_TRAIN = "final_model.zip"

# -------------------
# HARD MODE FINE-TUNE
# -------------------
LOAD_FROM = r"runs\rppo_sepLSTM_heur_20251122_210239\fine_tune_model_v2"
FINE_TUNE_STEPS = 300_000   # sadece hard mode için ek eğitim adımı
NEW_LR = 1e-4               # fine-tune için biraz daha küçük learning rate
SAVE_NAME_TUNED = "fine_tune_model_v3.zip"


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
def evaluate_win_rate(model, vec_env_norm: VecNormalize, episodes=100):
    """
    VecNormalize + (SubprocVecEnv veya DummyVecEnv) üzerinde
    LSTM'li RecurrentPPO için win-rate hesaplar.

    - Aynı VecNormalize'ı kullanır (obs istatistikleri tutarlı kalır)
    - env.training bayrağını geçici olarak False yapar (istatistikler bozulmaz)
    - Tüm biten epizotları sayar; winner == 0 ise win++.
    """
    env = vec_env_norm
    n_envs = env.num_envs

    # VecNormalize istatistiklerini dondur
    old_training_flag = getattr(env, "training", None)
    if hasattr(env, "training"):
        env.training = False

    # Başlangıç reset
    obs = env.reset()
    lstm_states = None
    episode_starts = np.ones(n_envs, dtype=bool)

    wins = 0
    total_eps = 0

    while total_eps < episodes:
        actions, lstm_states = model.predict(
            obs,
            state=lstm_states,
            episode_start=episode_starts,
            deterministic=True,
        )

        obs, rewards, dones, infos = env.step(actions)
        episode_starts = dones

        for i, done in enumerate(dones):
            if done:
                info = infos[i]
                winner = info.get("winner", -1)

                # Sadece bizim ajan kazandıysa win++
                if winner == 0:
                    wins += 1

                # Ama berabere / kayıp da olsa bu bir epizottur → say
                total_eps += 1

                if total_eps >= episodes:
                    break

    # training bayrağını eski haline getir
    if old_training_flag is not None:
        env.training = old_training_flag

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

    if LOAD_PRESENT_MODEL:
        # HARD MODE FINE-TUNE
        print(f"[FINE-TUNE] Loading model from: {LOAD_FROM}")
        model = RecurrentPPO.load(LOAD_FROM, env=train_vecnorm)

        # fine-tune hyperparam güncelle (isteğe bağlı ama tavsiye)
        model.learning_rate = NEW_LR  # 1e-4
        model.ent_coef = 0.003        # biraz daha düşük entropy
        reset_flag = False
        total_steps = FINE_TUNE_STEPS  # 300k
    else:
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

    if LOAD_PRESENT_MODEL:
        model.learn(
            total_timesteps=total_steps,
            reset_num_timesteps=reset_flag,
        )
        save_name = SAVE_NAME_TUNED
    else:
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
            save_name = SAVE_NAME_TRAIN

    # --- Final kayıt ---
    final_model_path = os.path.join(log_dir, save_name)
    model.save(final_model_path)
    train_vecnorm.save(os.path.join(log_dir, "vecnormalize.pkl"))
    print(f"[INFO] Final model saved: {final_model_path}")

    # --- Geniş değerlendirme ---
    wr = evaluate_win_rate(model, train_vecnorm, episodes=EVAL_EPISODES)
    print(f"[EVAL] Win-rate vs HEURISTIC ({EVAL_EPISODES} ep): {wr:.2%}")
    print(f"[TENSORBOARD] tensorboard --logdir {log_dir}")

    # --- Replay ---
    save_path = os.path.join(log_dir, "replay_heur.gif")  # .gif de verebilirsin

    def _make_env():
        # aynı heuristic-rakipli wrapper
        return DogfightSoloEnvPool(seed=123, opponent_pool=['heuristic'], pool_probs=[1.0])

    info_last = render_eval_episode(
        model=model,
        make_env_fn=_make_env,
        save_path=save_path,
        vecnorm=train_vecnorm,  # eğitimde kullandığın VecNormalize’ı geç
        fps=30,
        max_steps=1200,
        deterministic=True,
    )
    print("[REPLAY] saved to:", save_path, "| info:", info_last)
