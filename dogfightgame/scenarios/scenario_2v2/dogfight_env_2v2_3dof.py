import os
import numpy as np
from gymnasium import spaces
from scenarios.scenario_2v2.heuristic_bot_3dof import HeuristicBot3DOF


# =========================================================
# 2v2 3DOF Dogfight Environment
# =========================================================
class Dogfight2v2_3DOF:
    def __init__(self, dt=0.1, arena=6000.0, seed=0):
        self.dt = dt
        self.arena = arena
        self.rng = np.random.default_rng(seed)

        # ---- Physics ----
        self.g = 9.81
        self.v_min = 120.0
        self.v_max = 350.0
        self.gamma_max = np.deg2rad(45)

        # ---- Episode control ----
        self.max_steps = 3000  # <<< TIME LIMIT (kritik)

        # ---- Teams ----
        self.n_agents = 4
        self.team_A = [0, 1]
        self.team_B = [2, 3]

        # ---- Action: [nx, nz, mu, fire] ----
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0, -np.deg2rad(60), 0.0], dtype=np.float32),
            high=np.array([1.5, 9.0,  np.deg2rad(60), 1.0], dtype=np.float32),
        )

        # ---- Observation ----
        obs_dim = 23
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )

        # ---- Weapon / WEZ ----
        self.wez_R = 900.0
        self.wez_ang = np.deg2rad(25.0)
        self.lead_gate = np.deg2rad(20.0)
        self.bullet_speed = 300.0
        self.base_pk = 0.55

        # ---- Ammo ----
        self.initial_ammo = 80
        self.fire_cd_steps = 3

        # ---- Reward ----
        self.track_w = 0.02
        self.shot_w = 0.01
        self.kill_w = 1.0
        self.bad_shot_w = 0.01

        # ---- Opponent ----
        self.heuristic_bot = HeuristicBot3DOF()

        # ---- Logging ----
        self.enable_logging = False
        self.log_dir = "runs/replays_tmp"
        self.episode_id = 0

        self.reset()

    # -----------------------------------------------------

    def reset(self):
        self.episode_id += 1
        self.steps = 0

        self.s = np.zeros((self.n_agents, 6), dtype=np.float32)
        for i in range(self.n_agents):
            ang = self.rng.uniform(0, 2 * np.pi)
            rad = self.rng.uniform(1500, 2500)
            self.s[i, 0] = rad * np.cos(ang)
            self.s[i, 1] = rad * np.sin(ang)
            self.s[i, 2] = self.rng.uniform(1000, 2000)
            self.s[i, 3] = self.rng.uniform(180, 240)
            self.s[i, 4] = self.rng.uniform(-np.pi, np.pi)
            self.s[i, 5] = 0.0

        self.hp = np.ones(self.n_agents, dtype=np.float32)
        self.ammo = np.full(self.n_agents, self.initial_ammo, dtype=np.int32)
        self.fire_cd = np.zeros(self.n_agents, dtype=np.int32)

        self._traj = {"state": [], "action": [], "reward": [], "hp": [], "ammo": []}
        return self._obs_all()

    # -----------------------------------------------------

    def _integrate_3dof(self, i, nx, nz, mu):
        x, y, h, V, psi, gamma = self.s[i]

        dx = V * np.cos(psi) * np.cos(gamma)
        dy = V * np.sin(psi) * np.cos(gamma)
        dh = V * np.sin(gamma)

        dV = self.g * (nx - np.sin(gamma))
        dpsi = self.g * nz * np.sin(mu) / max(V * np.cos(gamma), 1e-3)
        dgamma = self.g * (nz * np.cos(mu) - np.cos(gamma)) / max(V, 1e-3)

        x += dx * self.dt
        y += dy * self.dt
        h += dh * self.dt
        V = np.clip(V + dV * self.dt, self.v_min, self.v_max)
        psi = (psi + dpsi * self.dt + np.pi) % (2 * np.pi) - np.pi
        gamma = np.clip(gamma + dgamma * self.dt, -self.gamma_max, self.gamma_max)

        self.s[i] = [x, y, h, V, psi, gamma]

    # -----------------------------------------------------

    def _rel_geom(self, i, j):
        dx = self.s[j, 0] - self.s[i, 0]
        dy = self.s[j, 1] - self.s[i, 1]
        R = np.hypot(dx, dy)

        psi_i = self.s[i, 4]
        psi_j = self.s[j, 4]

        brg = (np.arctan2(dy, dx) - psi_i + np.pi) % (2 * np.pi) - np.pi
        aoff = (psi_j - psi_i + np.pi) % (2 * np.pi) - np.pi

        vij = self.s[j, 3] * np.array([np.cos(psi_j), np.sin(psi_j)])
        vii = self.s[i, 3] * np.array([np.cos(psi_i), np.sin(psi_i)])
        clos = -(dx * (vij[0] - vii[0]) + dy * (vij[1] - vii[1])) / (R + 1e-6)

        return R, brg, aoff, clos

    def _lead_error(self, i, j):
        dx = self.s[j, 0] - self.s[i, 0]
        dy = self.s[j, 1] - self.s[i, 1]
        psi_i = self.s[i, 4]
        psi_j = self.s[j, 4]
        vj = self.s[j, 3]

        R = np.hypot(dx, dy)
        t_hit = R / (self.bullet_speed + 1e-6)

        lead_x = dx + vj * np.cos(psi_j) * t_hit
        lead_y = dy + vj * np.sin(psi_j) * t_hit
        lead_brg = (np.arctan2(lead_y, lead_x) - psi_i + np.pi) % (2 * np.pi) - np.pi
        return abs(lead_brg)

    # -----------------------------------------------------

    def _obs_agent(self, i):
        obs = []
        V = self.s[i, 3]
        psi = self.s[i, 4]
        gamma = self.s[i, 5]

        obs.extend([
            (V - self.v_min) / (self.v_max - self.v_min),
            np.sin(psi), np.cos(psi),
            np.sin(gamma), np.cos(gamma),
        ])

        teammates = [j for j in self.team_A if j != i]
        enemies = self.team_B

        for j in teammates + enemies:
            R, brg, aoff, clos = self._rel_geom(i, j)
            obs.extend([
                np.clip(R / 4000.0, 0.0, 1.0),
                np.sin(brg), np.cos(brg),
                np.sin(aoff), np.cos(aoff),
                np.tanh(clos / 200.0),
            ])

        return np.clip(np.array(obs, dtype=np.float32), -1.0, 1.0)

    def _obs_all(self):
        return {i: self._obs_agent(i) for i in self.team_A}

    # -----------------------------------------------------

    def step(self, actions):
        self.steps += 1
        reward = {i: 0.0 for i in self.team_A}
        done = False
        info = {}

        for i in range(self.n_agents):
            if self.fire_cd[i] > 0:
                self.fire_cd[i] -= 1

        for i in self.team_A:
            nx, nz, mu, _ = actions[i]
            self._integrate_3dof(i, nx, nz, mu)

        for i in self.team_B:
            obs_i = self._obs_agent(i)
            nx, nz, mu, _ = self.heuristic_bot.act(obs_i)
            self._integrate_3dof(i, nx, nz, mu)

        for i in self.team_A:
            for j in self.team_B:
                if self.hp[j] <= 0:
                    continue

                R, brg, _, _ = self._rel_geom(i, j)
                lead_err = self._lead_error(i, j)

                inside_wez = (R < self.wez_R) and (abs(brg) < self.wez_ang)
                lead_ok = lead_err < self.lead_gate

                reward[i] += self.track_w * np.cos(lead_err)
                reward[i] += -0.0001 * R

                fire_p = actions[i][3]
                fire_cmd = (
                    fire_p > 0.5
                    and inside_wez
                    and self.fire_cd[i] == 0
                    and self.ammo[i] > 0
                )

                if fire_cmd:
                    self.ammo[i] -= 1
                    self.fire_cd[i] = self.fire_cd_steps

                    if lead_ok:
                        pk = self.base_pk * np.exp(-(lead_err / self.lead_gate) ** 2)
                        pk = float(np.clip(pk, 0.1, 0.9))
                        reward[i] += self.shot_w

                        if self.rng.random() < pk:
                            self.hp[j] -= 1.0
                            reward[i] += self.kill_w
                    else:
                        reward[i] -= self.bad_shot_w

        # ---------- TIME LIMIT TERMINATION ----------
        if self.steps >= self.max_steps and not done:
            done = True
            info["winner"] = "B"
            info["termination"] = "time_limit"
            for i in self.team_A:
                reward[i] -= 0.5

        if self.enable_logging:
            self._traj["state"].append(self.s.copy())
            self._traj["action"].append(actions.copy())
            self._traj["reward"].append(reward.copy())
            self._traj["hp"].append(self.hp.copy())
            self._traj["ammo"].append(self.ammo.copy())

        if done and self.enable_logging:
            os.makedirs(self.log_dir, exist_ok=True)
            path = os.path.join(self.log_dir, f"episode_{self.episode_id:05d}.npz")
            np.savez_compressed(
                path,
                state=np.array(self._traj["state"]),
                action=np.array(self._traj["action"]),
                reward=np.array(self._traj["reward"]),
                hp=np.array(self._traj["hp"]),
                ammo=np.array(self._traj["ammo"]),
            )

        return self._obs_all(), reward, done, info
