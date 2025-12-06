# dogfight_wrappers.py
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from dogfight_env import Dogfight1v1


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
# 2) Rakip havuzlu tek-ajan Gym wrapper (continuous)
# ----------------------------------------------------
class DogfightSoloEnvPool(gym.Env):
    """
    - Tek kontrol edilen ajan: 0
    - Ajan 1 heuristic_continuous ile uçuyor.
    - Observation/action space, DOĞRUDAN Dogfight1v1'den alınır
      (obs_dim'i tekrar hesaplamıyoruz!).
    """
    metadata = {"render_modes": []}

    def __init__(self, seed: int = 0, opponent_pool=None, pool_probs=None):
        super().__init__()
        self.base = Dogfight1v1(seed=seed)

        # ---- Space'ler: tek kaynak Dogfight1v1 ----
        self.observation_space = self.base.observation_space
        self.action_space = self.base.action_space

        # Opponent pool — şimdilik sadece heuristic
        self.opponent_pool = opponent_pool or ["heuristic"]
        self.pool_probs = pool_probs
        self._opponent_type = "heuristic"

    # ----- opponent seç (ileride pool genişlerse) -----
    def _choose_opponent_spec(self):
        return "heuristic"

    # ----- opponent aksiyonu -----
    def _opponent_act(self, obs_vec):
        return heuristic_policy_continuous(obs_vec)

    # ----- Gym API -----
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            # Dogfight1v1 kendi RNG'sini kullanıyor
            pass

        self._opponent_type = self._choose_opponent_spec()
        obs_dict = self.base.reset()
        self._last_info = {}
        # SB3: (obs, info)
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
