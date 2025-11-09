# train_ippo_pool.py
# IPPO (PPO) + Opponent Pool (heuristic + snapshot'lar)
# - SubprocVecEnv + VecNormalize + VecMonitor
# - Entropy annealing
# - Havuz: ['heuristic', 'path/to/snapshot.zip', ...]
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

from dogfight_env import Dogfight1v1   # ← senin güncel env'in

# ----------------------------------------------------
# 1) Heuristic bot (değişmedi)
# ----------------------------------------------------
def heuristic_policy_continuous(obs_vec):
    # obs: [R_norm, sin(brg), cos(brg), sin(aoff), cos(aoff), clos_norm, v_norm, sin(bank), cos(bank), ammo, (boost?)]
    R = 2000.0 * float(np.clip(obs_vec[0], 0, 1))
    brg = float(np.arctan2(obs_vec[1], obs_vec[2]))  # sin/cos -> açı
    aoff = float(np.arctan2(obs_vec[3], obs_vec[4]))
    # kaba yönlendirme: bearing işareti kadar yat
    eps = np.deg2rad(3)
    bank_rate = 0.0
    if brg > eps: bank_rate = +1.0
    if brg < -eps: bank_rate = -1.0

    # hız: kapanma (clos_norm) negatifse yaklaş, pozitifse yavaşla
    clos = 200.0 * np.arctanh(np.clip(obs_vec[5], -0.999, 0.999))
    throttle = 1.0 if clos < 0 else (0.0 if clos > 120 else 0.5)

    # ateş: gevşek WEZ
    inside_wez = (R < 800.0) and (abs(brg) < np.deg2rad(25)) and (abs(aoff) < np.deg2rad(20))
    trigger_p = 1.0 if inside_wez else 0.0
    return np.array([bank_rate, throttle, trigger_p], dtype=np.float32)

# ----------------------------------------------------
# 2) Rakip havuzlu tek-ajan Gym wrapper
#    - Opponent pool: ['heuristic', '/path/snapshot1.zip', ...]
#    - Snapshot modelleri env içinde cache'lenir; reset'te rastgele seçilir
# ----------------------------------------------------
class DogfightSoloEnvPool(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, seed=0, opponent_pool=None, pool_probs=None):
        super().__init__()
        self.base = Dogfight1v1(seed=seed)

        # --- Observation & Action Space ---
        # boost varsa 11, yoksa 10 boyutlu vektör
        obs_dim = 10
        if hasattr(self.base, "boost_energy_max"):
            obs_dim += 1

        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(obs_dim,),
            dtype=np.float32
        )

        # Aksiyonlar: [bank_cmd, throttle_cmd, fire_cmd]
        # (Henüz continuous değilse)
        # self.action_space = spaces.MultiDiscrete([3, 3, 2])
        # Eğer continuous moda geçtiysek yukarıdaki satır yerine şunu kullanırız:
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([+1.0, 1.0, 1.0], dtype=np.float32)
        )

        # Opponent pool (değişmedi)
        self.opponent_pool = opponent_pool or ['heuristic']
        self.pool_probs = pool_probs
        self._opponent_type = 'heuristic'
        self._opponent_model_cache = {}
        self._ppo_cls = PPO

    # ----- public API: dışarıdan havuzu güncelle -----
    def set_opponent_pool(self, pool_list, pool_probs=None):
        self.opponent_pool = list(pool_list)
        if pool_probs is None:
            self.pool_probs = None
        else:
            p = np.array(pool_probs, dtype=np.float64)
            self.pool_probs = (p / p.sum()).tolist()

    # ----- opponent seç -----
    def _choose_opponent_spec(self):
        if self.pool_probs is None or len(self.pool_probs) != len(self.opponent_pool):
            return random.choice(self.opponent_pool)
        return random.choices(self.opponent_pool, weights=self.pool_probs, k=1)[0]

    # ----- opponent aksiyonu -----
    def _opponent_act(self, obs_vec):
        if self._opponent_type == 'heuristic':
            return heuristic_policy_continuous(obs_vec)
        else:
           """ # SB3 snapshot modeli kullan
            model = self._opponent_model_cache.get(self._opponent_type, None)
            if model is None:
                # .zip'i yükle ve cache'e koy
                model = self._ppo_cls.load(self._opponent_type)
                self._opponent_model_cache[self._opponent_type] = model
            # Snapshot policy, single obs bekler; bizim obs_vec tek agent -> (1,7) shape
            a, _ = model.predict(obs_vec, deterministic=True)
            # SB3 MultiDiscrete aksiyonunu düzleştir (np.array [3], int'lere çevir)
            a = np.asarray(a).astype(int).ravel()
            return (int(a[0]), int(a[1]), int(a[2]))"""
           return heuristic_policy_continuous(obs_vec)

    # ----- Gym API -----
    def _tuple_from_action(self, a):
        a = np.asarray(a).astype(int)
        return (int(a[0]), int(a[1]), int(a[2]))

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            # base env kendi RNG'sini kullanıyor; burada dokunmuyoruz
            pass
        # Bu epizod için rakibi seç
        spec = self._choose_opponent_spec()
        self._opponent_type = spec  # 'heuristic' veya '/path/to/model.zip'
        obs_dict = self.base.reset()
        self._last_info = {}
        return obs_dict[0], {}

    def step(self, action):
        a0 = self._tuple_from_action(action)
        obs_dict = self.base._obs_all()
        a1 = self._opponent_act(obs_dict[1])  # havuzdan seçilen rakip
        obs_next, r, done, info = self.base.step({0: a0, 1: a1})
        self._last_info = info
        return obs_next[0], float(r[0]), done, False, info


# ----------------------------------------------------
# 3) Değerlendirme (win-rate) – eğitim VecNormalize ile uyumlu
# ----------------------------------------------------
def evaluate_trained(model, train_vecnorm, episodes=200, seed=4242, opponent_pool=None):
    if opponent_pool is None:
        opponent_pool = ['heuristic']

    def make_one():
        return DogfightSoloEnvPool(seed=seed, opponent_pool=opponent_pool)

    eval_env = DummyVecEnv([make_one])
    eval_env = VecMonitor(eval_env, filename=None)
    eval_env = VecNormalize(eval_env, training=False, norm_obs=True, norm_reward=False)
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

    eval_env.close()
    return wins / episodes


# ----------------------------------------------------
# 4) main – eğitim + snapshot ekleme + havuz güncelleme
# ----------------------------------------------------
if __name__ == "__main__":
    run_name = time.strftime("ppo_ippo_pool_%Y%m%d_%H%M%S")
    log_dir = os.path.join("runs", run_name)
    os.makedirs(log_dir, exist_ok=True)

    # Başlangıç havuzu: sadece heuristic
    opponent_pool = ['heuristic']

    # --------- Paralel eğitim ortamı (havuz destekli) ---------
    N_ENVS = 8
    def make_env(rank):
        def _f():
            return DogfightSoloEnvPool(seed=10_000 + rank, opponent_pool=opponent_pool)
        return _f

    train_env = SubprocVecEnv([make_env(i) for i in range(N_ENVS)])
    train_env = VecMonitor(train_env, filename=None)
    train_vecnorm = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=5.0)

    # Logger
    new_logger = configure(log_dir, ["stdout", "tensorboard"])

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
        ent_coef=0.02,    # başlayınca yüksek; annealing ile düşüreceğiz
        vf_coef=0.5,
        verbose=1,
        tensorboard_log=log_dir,
        seed=42,
    )
    model.set_logger(new_logger)

    # --------- Eval env (light parallel) ---------
    def make_eval_env(rank=0):
        def _f():
            return DogfightSoloEnvPool(seed=99_000 + rank, opponent_pool=opponent_pool)
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

    # --------- Eğitim & Snapshot planı ---------
    # Milestones: adımlarda snapshot kaydet ve havuza ekle
    milestones = [200_000, 400_000, 600_000, 800_000]
    entropy_schedule = {200_000: 0.012, 500_000: 0.008, 800_000: 0.005}

    done_steps = 0
    t0 = time.time()

    for target in milestones:
        chunk = target - done_steps
        model.learn(total_timesteps=chunk, reset_num_timesteps=False, callback=eval_callback)
        done_steps = target

        # 1) Snapshot kaydet
        snap_path = os.path.join(log_dir, f"snapshot_{done_steps}.zip")
        model.save(snap_path)
        print(f"[POOL] Snapshot saved: {snap_path}")

        # 2) Havuzu güncelle (heuristic + yeni snapshotlar)
        if snap_path not in opponent_pool:
            opponent_pool.append(snap_path)
        print(f"[POOL] New pool: {opponent_pool}")

        # 3) Tüm eğitim env'lerine havuzu bildir
        #    SubprocVecEnv: env_method ile alt süreçlerde set_opponent_pool çağır
        train_env.env_method("set_opponent_pool", opponent_pool)

        # 4) Eval env'lerini de güncelle
        raw_eval_env.env_method("set_opponent_pool", opponent_pool)

        # 5) Entropy annealing (milestone'a göre)
        if target in entropy_schedule:
            new_ent = entropy_schedule[target]
            model.ent_coef = new_ent
            print(f"[ANNEAL] ent_coef -> {new_ent}")

    # --------- Final kaydet ---------
    final_model_path = os.path.join(log_dir, "final_model.zip")
    model.save(final_model_path)
    train_vecnorm.save(os.path.join(log_dir, "vecnormalize.pkl"))
    print(f"[INFO] Final model saved: {final_model_path}")

    # --------- Geniş değerlendirme ---------
    wr_heur = evaluate_trained(model, train_vecnorm, episodes=200, opponent_pool=['heuristic'])
    wr_pool = evaluate_trained(model, train_vecnorm, episodes=200, opponent_pool=opponent_pool)
    print(f"[EVAL] Win-rate vs HEURISTIC (200 ep): {wr_heur:.2%}")
    print(f"[EVAL] Win-rate vs POOL (200 ep): {wr_pool:.2%}")
    print(f"[TENSORBOARD] tensorboard --logdir {log_dir}")
