# =========================
# dogfight_wrappers.py
# =========================
import os
import pickle
import numpy as np
import gymnasium as gym

from dogfight_env import Dogfight1v1

# Expert opponent: RecurrentPPO
from sb3_contrib import RecurrentPPO


# ----------------------------------------------------
# 1) Heuristic bot (CONTINUOUS)
# ----------------------------------------------------
def heuristic_policy_continuous(obs_vec: np.ndarray) -> np.ndarray:
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
    if brg > eps:
        bank_rate = +1.0
    elif brg < -eps:
        bank_rate = -1.0

    # closure'a göre throttle
    if clos < 0:
        throttle = 1.0
    elif clos > 120:
        throttle = 0.0
    else:
        throttle = 0.5

    # gevşetilmiş WEZ (env içindeki WEZ'ten bağımsız, sadece bot için)
    inside_wez = (R < 900.0) and (abs(brg) < np.deg2rad(30)) and (abs(aoff) < np.deg2rad(25))
    trigger_p = 1.0 if inside_wez else 0.0

    return np.array([bank_rate, throttle, trigger_p], dtype=np.float32)


# ----------------------------------------------------
# 1.5) VecNormalize stats loader (pickle)
# ----------------------------------------------------
def _load_vecnormalize_obs_stats(vec_path: str):
    """
    SB3 VecNormalize.save() ile kaydedilen vecnormalize.pkl'den
    obs_rms mean/var ve clip_obs değerlerini okumaya çalışır.

    Farklı SB3 sürümlerinde pickle yapısı az da olsa değişebiliyor;
    bu fonksiyon olabildiğince toleranslı.
    """
    if not os.path.exists(vec_path):
        raise FileNotFoundError(f"vecnormalize.pkl not found: {vec_path}")

    with open(vec_path, "rb") as f:
        obj = pickle.load(f)

    # Bazı sürümlerde doğrudan VecNormalize objesi olarak gelir
    # Bazılarında dict/Namespace benzeri olabilir.
    clip_obs = getattr(obj, "clip_obs", None)

    obs_rms = getattr(obj, "obs_rms", None)
    if obs_rms is None and isinstance(obj, dict):
        obs_rms = obj.get("obs_rms", None)
        clip_obs = obj.get("clip_obs", clip_obs)

    if obs_rms is None:
        raise ValueError("Could not find obs_rms inside vecnormalize.pkl")

    mean = getattr(obs_rms, "mean", None)
    var = getattr(obs_rms, "var", None)

    # bazı durumlarda dict olabilir
    if mean is None and isinstance(obs_rms, dict):
        mean = obs_rms.get("mean", None)
        var = obs_rms.get("var", None)

    if mean is None or var is None:
        raise ValueError("Could not extract obs_rms.mean/var from vecnormalize.pkl")

    if clip_obs is None:
        # SB3 default
        clip_obs = 10.0

    return np.array(mean, dtype=np.float32), np.array(var, dtype=np.float32), float(clip_obs)


class _FrozenExpertOpponent:
    """
    Frozen RecurrentPPO opponent:
    - model: RecurrentPPO.load(model_path)
    - obs normalization: vecnormalize.pkl içinden obs_rms mean/var + clip_obs
    - LSTM state yönetimi (episode_start)
    """
    def __init__(self, model_path: str, vec_path: str, device: str = "cpu"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Expert model not found: {model_path}")
        self.model_path = model_path
        self.vec_path = vec_path
        self.device = device

        self.model = RecurrentPPO.load(model_path, device=device)
        self.mean, self.var, self.clip_obs = _load_vecnormalize_obs_stats(vec_path)
        self.eps = 1e-8

        self.lstm_state = None
        self.episode_start = True

    def reset(self):
        self.lstm_state = None
        self.episode_start = True

    def _norm_obs(self, obs: np.ndarray) -> np.ndarray:
        # obs: (obs_dim,)
        obs = obs.astype(np.float32)
        obs_n = (obs - self.mean) / np.sqrt(self.var + self.eps)
        obs_n = np.clip(obs_n, -self.clip_obs, self.clip_obs)
        return obs_n

    def act(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        obs_n = self._norm_obs(obs)
        # RecurrentPPO predict genelde (n_env, obs_dim) bekler, güvenli olsun:
        obs_in = obs_n.reshape(1, -1)
        action, self.lstm_state = self.model.predict(
            obs_in,
            state=self.lstm_state,
            episode_start=np.array([self.episode_start], dtype=bool),
            deterministic=deterministic,
        )
        self.episode_start = False
        # action: (1, act_dim) veya (act_dim,)
        action = np.asarray(action).reshape(-1)
        return action.astype(np.float32)


# ----------------------------------------------------
# 2) Rakip havuzlu tek-ajan Gym wrapper (continuous)
# ----------------------------------------------------
class DogfightSoloEnvPool(gym.Env):
    """
    - Tek kontrol edilen ajan: 0
    - Ajan 1: heuristic_continuous veya frozen expert policy.
    - Observation/action space, DOĞRUDAN Dogfight1v1'den alınır.
    """
    metadata = {"render_modes": []}

    def __init__(
        self,
        seed: int = 0,
        opponent_pool=None,
        pool_probs=None,
        timeout_as_loss: bool = False,
        expert_device: str = "cpu",
    ):
        super().__init__()
        self.base = Dogfight1v1(seed=seed, timeout_as_loss=timeout_as_loss)

        # ---- Space'ler: tek kaynak Dogfight1v1 ----
        self.observation_space = self.base.observation_space
        self.action_space = self.base.action_space

        # Opponent pool:
        # - "heuristic"
        # - ("expert", "path/to/model.zip", "path/to/vecnormalize.pkl")
        self.opponent_pool = opponent_pool or ["heuristic"]
        self.pool_probs = pool_probs

        if self.pool_probs is None:
            # eşit olasılık
            self.pool_probs = [1.0 / len(self.opponent_pool)] * len(self.opponent_pool)

        self.pool_probs = np.array(self.pool_probs, dtype=np.float32)
        self.pool_probs = self.pool_probs / (self.pool_probs.sum() + 1e-8)

        self.rng = np.random.default_rng(seed + 12345)
        self._opponent_spec = "heuristic"
        self._expert = None
        self._expert_device = expert_device

    # ----- opponent seç (pool genişlerse) -----
    def _choose_opponent_spec(self):
        idx = int(self.rng.choice(len(self.opponent_pool), p=self.pool_probs))
        return self.opponent_pool[idx]

    def _ensure_expert_loaded(self, model_path: str, vec_path: str):
        if self._expert is None:
            self._expert = _FrozenExpertOpponent(
                model_path=model_path,
                vec_path=vec_path,
                device=self._expert_device,
            )

    # ----- opponent aksiyonu -----
    def _opponent_act(self, obs_vec: np.ndarray) -> np.ndarray:
        spec = self._opponent_spec

        # "heuristic"
        if isinstance(spec, str):
            return heuristic_policy_continuous(obs_vec)

        # ("expert", model_path, vec_path)
        if isinstance(spec, (tuple, list)) and len(spec) >= 3:
            opp_type = spec[0]
            if opp_type != "expert":
                # bilinmeyen → heuristic
                return heuristic_policy_continuous(obs_vec)

            model_path = spec[1]
            vec_path = spec[2]
            self._ensure_expert_loaded(model_path, vec_path)
            return self._expert.act(obs_vec, deterministic=True)

        # fallback
        return heuristic_policy_continuous(obs_vec)

    # ----- Gym API -----
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            # Dogfight1v1 kendi RNG'sini kullanıyor; burada ekstra seedlemek istemiyoruz
            pass

        self._opponent_spec = self._choose_opponent_spec()

        # expert seçildiyse episode başlat
        if isinstance(self._opponent_spec, (tuple, list)) and len(self._opponent_spec) >= 3:
            if self._opponent_spec[0] == "expert":
                self._ensure_expert_loaded(self._opponent_spec[1], self._opponent_spec[2])
                self._expert.reset()

        obs_all = self.base.reset()
        self._last_info = {}
        # SB3: (obs, info)
        return obs_all[0], {}

    def step(self, action):
        # continuous aksiyonu doğrudan aktar
        a0 = np.asarray(action, dtype=np.float32).ravel()

        # rakip aksiyonu (agent1 obs)
        obs_all = self.base._obs_all()
        a1 = self._opponent_act(obs_all[1])

        # base.step dict bekliyor (senin mevcut env yapın)
        obs_next, r, done, info = self.base.step({0: a0, 1: a1})
        self._last_info = info

        # Gymnasium sözleşmesi: (obs, reward, terminated, truncated, info)
        terminated = bool(done)
        truncated = False
        return obs_next[0], float(r[0]), terminated, truncated, info
