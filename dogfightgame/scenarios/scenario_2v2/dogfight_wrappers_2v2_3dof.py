import gymnasium as gym
import numpy as np

from scenarios.scenario_2v2.dogfight_env_2v2_3dof import Dogfight2v2_3DOF


class Dogfight2v2SB3Wrapper(gym.Env):
    """
    SB3 uyumlu 2v2 wrapper (centralized execution):

    - Kontrol edilen ajanlar: Team A = [0, 1]
    - Tek policy
    - Observation: concat(obs_0, obs_1)
    - Action: [nx0, nz0, mu0, fire0, nx1, nz1, mu1, fire1]
    - Reward: team reward
    """

    metadata = {"render_modes": []}

    def __init__(self, env: Dogfight2v2_3DOF):
        super().__init__()
        self.env = env

        # ---- Action space ----
        # 2 agent × [nx, nz, mu, fire]
        low = np.tile(env.action_space.low, 2).astype(np.float32)
        high = np.tile(env.action_space.high, 2).astype(np.float32)

        self.action_space = gym.spaces.Box(
            low=low,
            high=high,
            dtype=np.float32,
        )

        # ---- Observation space ----
        obs_dim = env.observation_space.shape[0]
        self.observation_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2 * obs_dim,),
            dtype=np.float32,
        )

    # -------------------------------------------------

    def reset(self, *, seed=None, options=None):
        obs_dict = self.env.reset()

        obs = np.concatenate(
            [obs_dict[0], obs_dict[1]],
            axis=0,
        )

        info = {}
        return obs, info

    # -------------------------------------------------

    def step(self, action):
        """
        action shape: (8,)
        [nx0, nz0, mu0, fire0, nx1, nz1, mu1, fire1]
        """
        action = np.asarray(action, dtype=np.float32).reshape(-1)

        a0 = action[0:4]
        a1 = action[4:8]

        obs_dict, reward_dict, done, info = self.env.step(
            {
                0: a0,
                1: a1,
            }
        )

        obs = np.concatenate(
            [obs_dict[0], obs_dict[1]],
            axis=0,
        )

        # ---- Team reward ----
        reward = float(
            reward_dict.get(0, 0.0) + reward_dict.get(1, 0.0)
        )

        terminated = bool(done)
        truncated = False

        return obs, reward, terminated, truncated, info
