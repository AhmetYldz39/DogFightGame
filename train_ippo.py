# train_ippo.py
# IPPO (PPO) ile 1v1 dogfight eğitimi
# - Rakip: heuristic bot
# - Paralel ortam (SubprocVecEnv)
# - VecNormalize (obs+reward)
# - VecMonitor (episodic reward/length logları için)
# - Entropy annealing (keşiften odaklanmaya)

import os
import time
import numpy as np
import gymnasium as gym
import torch.nn as nn

from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.logger import configure

from dogfight_env import Dogfight1v1

# ----------------------------------------------------
# Heuristic bot (rakip)
# ----------------------------------------------------
def heuristic_policy(obs_vec):
    """
    obs_vec: np.array([...])  # agent'ın egosantrik gözlemi
      [R_norm, brg_norm, aoff_norm, clos_norm, v_norm, bank_norm, ammo_norm]
    return: (bank_cmd, thr_cmd, fire)  -> (0..2, 0..2, 0..1)
    """
    brg = np.pi * obs_vec[1]
    aoff = np.pi * obs_vec[2]
    R = 2000.0 * obs_vec[0]

    # bank komutu (bearing'i sıfırlamaya çalış)
    if brg > np.deg2rad(3): bank_cmd = 2
    elif brg < -np.deg2rad(3): bank_cmd = 0
    else: bank_cmd = 1

    # throttle (closure'a göre kaba hız ayarı)
    clos = 200.0 * np.arctanh(np.clip(obs_vec[3], -0.999, 0.999))
    if clos < 0:      thr_cmd = 2
    elif clos > 120:  thr_cmd = 0
    else:             thr_cmd = 1

    # ateş (WEZ + küçük angle-off)
    inside_wez = (R < 600.0) and (abs(brg) < np.deg2rad(20))
    fire = 1 if (inside_wez and abs(aoff) < np.deg2rad(15)) else 0
    return (bank_cmd, thr_cmd, fire)

# ----------------------------------------------------
# Gym wrapper: Tek-ajan görünüm (agent-0 eğitilir)
# ----------------------------------------------------
class DogfightSoloEnv(gym.Env):
    """
    SB3 uyumlu tek-ajan env:
      - Öğretilen ajan: index 0
      - Rakip ajan: heuristic bot (index 1)
    Observation: agent-0 vektörü (float32, boyut=7)
    Action: MultiDiscrete([3, 3, 2])  -> (bank, throttle, fire)
    Reward: env.r[0]
    Done: env.done
    """
    metadata = {"render_modes": []}

    def __init__(self, seed=0):
        super().__init__()
        self.base = Dogfight1v1(seed=seed)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)
        self.action_space = spaces.MultiDiscrete([3, 3, 2])
        self._last_info = {}

    def _tuple_from_action(self, a):
        a = np.asarray(a).astype(int)
        return (int(a[0]), int(a[1]), int(a[2]))

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            # Basit tutuyoruz; taban env kendi RNG'sini kullanıyor
            pass
        obs_dict = self.base.reset()  # {0:vec, 1:vec}
        self._last_info = {}
        return obs_dict[0], {}

    def step(self, action):
        a0 = self._tuple_from_action(action)
        obs_dict = self.base._obs_all()
        a1 = heuristic_policy(obs_dict[1])

        obs_next, r, done, info = self.base.step({0: a0, 1: a1})
        self._last_info = info
        return obs_next[0], float(r[0]), done, False, info


# ----------------------------------------------------
# Değerlendirme (win-rate) - VecNormalize ile uyumlu
# ----------------------------------------------------
def evaluate_trained(model, train_vecnorm, episodes=100, seed=1234):
    # Tek ortamlı (1 env) değerlendirme için DummyVecEnv + VecNormalize
    def make_one():
        return DogfightSoloEnv(seed=seed)

    eval_env = DummyVecEnv([make_one])
    eval_env = VecMonitor(eval_env, filename=None)
    eval_env = VecNormalize(eval_env, training=False, norm_obs=True, norm_reward=False)

    # Eğitim env'inden obs istatistiklerini kopyala (şart!)
    eval_env.obs_rms = train_vecnorm.obs_rms

    wins = 0
    for _ in range(episodes):
        obs = eval_env.reset()
        done = False
        last_info = {}
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, infos = eval_env.step(action)
            done = bool(dones[0])
            if done:
                last_info = infos[0]
        if last_info.get("winner", None) == 0:
            wins += 1

    # Temizlik
    eval_env.close()
    return wins / episodes


# ----------------------------------------------------
# main
# ----------------------------------------------------
if __name__ == "__main__":
    run_name = time.strftime("ppo_ippo_%Y%m%d_%H%M%S")
    log_dir = os.path.join("runs", run_name)
    os.makedirs(log_dir, exist_ok=True)

    # ---------- Paralel eğitim ortamı ----------
    N_ENVS = 8

    def make_env(rank):
        def _f():
            # Her alt sürece farklı seed
            return DogfightSoloEnv(seed=10_000 + rank)
        return _f

    train_env = SubprocVecEnv([make_env(i) for i in range(N_ENVS)])
    # VecMonitor'ı VecNormalize'dan ÖNCE sarmalıyoruz (episodic reward/length ham veriden çıksın)
    train_env = VecMonitor(train_env, filename=None)
    # Normalizasyon: gözlem + ödül
    train_vecnorm = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=5.0)

    # ---------- SB3 logger ----------
    new_logger = configure(log_dir, ["stdout", "tensorboard"])

    # Policy mimarisi
    policy_kwargs = dict(
        net_arch=dict(pi=[128, 128], vf=[128, 128]),
        activation_fn=nn.Tanh,
    )

    model = PPO(
        policy="MlpPolicy",
        env=train_vecnorm,
        policy_kwargs=policy_kwargs,
        n_steps=512,                 # rollout horizon (efektif: N_ENVS * n_steps)
        batch_size=N_ENVS * 64,      # 512
        n_epochs=4,
        learning_rate=3e-4,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.02,               # başlangıç entropi
        vf_coef=0.5,
        verbose=1,
        tensorboard_log=log_dir,
        seed=42,
    )
    model.set_logger(new_logger)

    # ---------- Eval env (vector + normalize, training=False) ----------
    def make_eval_env(rank=0):
        def _f():
            return DogfightSoloEnv(seed=99_000 + rank)
        return _f

    raw_eval_env = SubprocVecEnv([make_eval_env(i) for i in range(2)])  # küçük paralel eval
    raw_eval_env = VecMonitor(raw_eval_env, filename=None)
    eval_vecnorm = VecNormalize(raw_eval_env, training=False, norm_obs=True, norm_reward=False)
    # Eğitim istatistiklerini eval'a kopyala
    eval_vecnorm.obs_rms = train_vecnorm.obs_rms

    eval_callback = EvalCallback(
        eval_vecnorm,
        best_model_save_path=log_dir,
        log_path=log_dir,
        eval_freq=20_000,            # her 20k adımda kısaca dene
        n_eval_episodes=10,
        deterministic=True,
        render=False,
    )

    # ---------- Entropy annealing ile parça parça eğitim ----------
    TOTAL_STEPS = 800_000
    schedule = [
        (200_000, 0.012),  # 0..200k: ent=0.02 (başlangıç) → 200k'dan sonra 0.012
        (500_000, 0.008),  # 200k..500k: 0.012 → 0.008
        (800_000, 0.005),  # 500k..800k: 0.008 → 0.005
    ]

    done = 0
    t0 = time.time()
    for milestone, new_ent in schedule:
        chunk = milestone - done
        model.learn(total_timesteps=chunk, reset_num_timesteps=False, callback=eval_callback)
        # entropi katsayısını güncelle
        model.ent_coef = new_ent
        done = milestone

    # ---------- Kaydet ----------
    model_path = os.path.join(log_dir, "final_model.zip")
    model.save(model_path)
    # VecNormalize istatistiklerini de kaydet (ileride yükeceksen lazım)
    train_vecnorm.save(os.path.join(log_dir, "vecnormalize.pkl"))

    print(f"[INFO] Model saved to: {model_path}")

    # ---------- Hızlı win-rate ölçümü (tek env ile 100 ep) ----------
    win_rate = evaluate_trained(model, train_vecnorm, episodes=100)
    print(f"[EVAL] Win-rate vs heuristic (100 ep): {win_rate:.2%}")
    print(f"[TENSORBOARD] Çalıştır: tensorboard --logdir {log_dir}")
