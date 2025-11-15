# train_ippo_pool.py
# PPO + Opponent Pool (şimdilik sadece HEURISTIC) + VecNormalize + VecMonitor
# Continuous action (Box): [bank_rate ∈ [-1,1], throttle ∈ [0,1], trigger_prob ∈ [0,1]]
#
# Çalıştırma:
#   python train_ippo_pool.py

import os
import time
import random
import numpy as np
import gymnasium as gym
import torch.nn as nn

from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.logger import configure

from dogfight_env import Dogfight1v1   # güncel continuous + sin/cos obs env'in

# ----------------------------------------------------
# 1) Heuristic bot (CONTINUOUS)
# ----------------------------------------------------
def heuristic_policy_continuous(obs_vec):
    """
    obs_vec: [R_norm, sin(brg), cos(brg), sin(aoff), cos(aoff),
              clos_norm, v_norm, sin(bank), cos(bank), ammo(, boost?)]
    """
    # temel metrikleri geri al
    R = 2000.0 * float(np.clip(obs_vec[0], 0.0, 1.0))
    brg = float(np.arctan2(obs_vec[1], obs_vec[2]))       # sin/cos -> açı
    aoff = float(np.arctan2(obs_vec[3], obs_vec[4]))
    clos = 200.0 * np.arctanh(np.clip(obs_vec[5], -0.999, 0.999))

    # bearing işaretine göre kaba yönlendirme
    eps = np.deg2rad(3)
    bank_rate = 0.0
    if brg >  eps: bank_rate = +1.0
    if brg < -eps: bank_rate = -1.0

    # closure'a göre throttle
    if clos < 0:
        throttle = 1.0
    elif clos > 120:
        throttle = 0.0
    else:
        throttle = 0.5

    # gevşetilmiş WEZ
    inside_wez = (R < 900.0) and (abs(brg) < np.deg2rad(30)) and (abs(aoff) < np.deg2rad(25))
    trigger_p = 1.0 if inside_wez else 0.0
    return np.array([bank_rate, throttle, trigger_p], dtype=np.float32)


# ----------------------------------------------------
# 2) Rakip havuzlu tek-ajan Gym wrapper (continuous)
# ----------------------------------------------------
class DogfightSoloEnvPool(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, seed=0, opponent_pool=None, pool_probs=None):
        super().__init__()
        self.base = Dogfight1v1(seed=seed)

        # --- Observation & Action Space ---
        # boost varsa 11, yoksa 10 boyutlu vektör
        obs_dim = 10 + (1 if hasattr(self.base, "boost_energy_max") else 0)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32)

        # Continuous action: [bank_rate, throttle, trigger_prob]
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([+1.0, 1.0, 1.0], dtype=np.float32)
        )

        # Opponent pool — şimdilik sadece heuristic (discrete snapshot'lar uyumsuz)
        self.opponent_pool = opponent_pool or ['heuristic']
        self.pool_probs = pool_probs
        self._opponent_type = 'heuristic'
        self._opponent_model_cache = {}
        self._ppo_cls = PPO  # ileride continuous snapshot yüklersen kullanılır

    # ----- public API: havuzu güncelle -----
    def set_opponent_pool(self, pool_list, pool_probs=None):
        # Continuous snapshot'lar hazır olana dek yalnızca heuristic'i izinli tutalım
        safe = ['heuristic']
        for x in (pool_list or []):
            if x == 'heuristic':
                safe = ['heuristic']
                break
        self.opponent_pool = safe
        if pool_probs is None:
            self.pool_probs = None
        else:
            p = np.array(pool_probs, dtype=np.float64)
            s = p.sum()
            self.pool_probs = (p / s).tolist() if s > 0 else None

    # ----- opponent seç -----
    def _choose_opponent_spec(self):
        # pratikte hep 'heuristic'
        return 'heuristic'

    # ----- opponent aksiyonu (continuous heuristic) -----
    def _opponent_act(self, obs_vec):
        return heuristic_policy_continuous(obs_vec)

    # ----- Gym API -----
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            # base env kendi RNG'sini kullanıyor
            pass
        self._opponent_type = self._choose_opponent_spec()
        obs_dict = self.base.reset()
        self._last_info = {}
        return obs_dict[0], {}

    def step(self, action):
        # continuous aksiyonu doğrudan aktar
        a0 = np.asarray(action, dtype=np.float32).ravel()
        # rakip aksiyonu
        obs_dict = self.base._obs_all()
        a1 = self._opponent_act(obs_dict[1])

        obs_next, r, done, info = self.base.step({0: a0, 1: a1})
        self._last_info = info
        # SB3 venv sözleşmesi: (obs, reward, terminated, truncated, info)
        return obs_next[0], float(r[0]), done, False, info


# ----------------------------------------------------
# 3) Değerlendirme (win-rate) – VecNormalize ile uyumlu
# ----------------------------------------------------
def evaluate_trained(model, train_vecnorm, episodes=200, seed=4242):
    def make_one():
        return DogfightSoloEnvPool(seed=seed, opponent_pool=['heuristic'], pool_probs=[1.0])

    eval_env = DummyVecEnv([make_one])
    eval_env = VecMonitor(eval_env, filename=None)
    eval_env = VecNormalize(eval_env, training=False, norm_obs=True, norm_reward=False)
    eval_env.obs_rms = train_vecnorm.obs_rms  # eğitim istatistiklerini kopyala

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

    eval_env.close()
    return wins / episodes


# ----------------------------------------------------
# 4) main – eğitim döngüsü (snapshot kapalı, entropi schedule açık)
# ----------------------------------------------------
if __name__ == "__main__":
    run_name = time.strftime("ppo_ippo_pool_cont_%Y%m%d_%H%M%S")
    log_dir = os.path.join("runs", run_name)
    os.makedirs(log_dir, exist_ok=True)

    # Başlangıç havuzu: sadece heuristic (continuous)
    opponent_pool = ['heuristic']

    # --------- Paralel eğitim ortamı ---------
    N_ENVS = 8
    def make_env(rank):
        def _f():
            return DogfightSoloEnvPool(seed=10_000 + rank, opponent_pool=opponent_pool, pool_probs=[1.0])
        return _f

    train_env = SubprocVecEnv([make_env(i) for i in range(N_ENVS)])
    train_env = VecMonitor(train_env, filename=None)
    train_vecnorm = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=5.0)

    # Logger
    logger = configure(log_dir, ["stdout", "tensorboard"])

    # Policy mimarisi
    policy_kwargs = dict(
        net_arch=dict(pi=[128, 128], vf=[128, 128]),
        activation_fn=nn.Tanh,
    )

    model = PPO(
        policy="MlpPolicy",
        env=train_vecnorm,
        policy_kwargs=policy_kwargs,
        n_steps=512,
        batch_size=N_ENVS * 64,
        n_epochs=4,
        learning_rate=3e-4,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.02,       # isterse 0.03 ile de başlayabilirsin
        vf_coef=0.5,
        verbose=1,
        tensorboard_log=log_dir,
        seed=42,
    )
    model.set_logger(logger)

    # --------- Eval env ---------
    def make_eval_env(rank=0):
        def _f():
            return DogfightSoloEnvPool(seed=99_000 + rank, opponent_pool=['heuristic'], pool_probs=[1.0])
        return _f

    raw_eval_env = SubprocVecEnv([make_eval_env(i) for i in range(2)])
    raw_eval_env = VecMonitor(raw_eval_env, filename=None)
    eval_vecnorm = VecNormalize(raw_eval_env, training=False, norm_obs=True, norm_reward=False)
    eval_vecnorm.obs_rms = train_vecnorm.obs_rms

    eval_callback = EvalCallback(
        eval_vecnorm,
        best_model_save_path=log_dir,
        log_path=log_dir,
        eval_freq=25_000,
        n_eval_episodes=10,
        deterministic=True,
        render=False,
    )

    # --------- Eğitim (snapshot yok) + entropi annealing ---------
    milestones = [200_000, 400_000, 600_000, 800_000]
    entropy_schedule = {200_000: 0.012, 500_000: 0.008, 800_000: 0.005}
    done_steps = 0

    for target in milestones:
        chunk = target - done_steps
        model.learn(total_timesteps=chunk, reset_num_timesteps=False, callback=eval_callback)
        done_steps = target

        # Entropy annealing (milestone'a göre)
        if target in entropy_schedule:
            model.ent_coef = entropy_schedule[target]
            print(f"[ANNEAL] ent_coef -> {model.ent_coef}")

        # Ara bilgi
        wr_tmp = evaluate_trained(model, train_vecnorm, episodes=100)
        print(f"[MILESTONE {done_steps}] WR vs HEUR(100): {wr_tmp:.2%}")

    # --------- Final kaydet ---------
    final_model_path = os.path.join(log_dir, "final_model.zip")
    model.save(final_model_path)
    train_vecnorm.save(os.path.join(log_dir, "vecnormalize.pkl"))
    print(f"[INFO] Final model saved: {final_model_path}")

    # --------- Geniş değerlendirme ---------
    wr_heur = evaluate_trained(model, train_vecnorm, episodes=200)
    print(f"[EVAL] Win-rate vs HEURISTIC (200 ep): {wr_heur:.2%}")
    print(f"[TENSORBOARD] tensorboard --logdir {log_dir}")
