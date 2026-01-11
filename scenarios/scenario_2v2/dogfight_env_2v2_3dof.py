import os
import numpy as np
from gymnasium import spaces
from scenarios.scenario_2v2.heuristic_bot_3dof import HeuristicBot3DOF


# =========================================================
# 2v2 3DOF Dogfight Environment (FINAL CLEAN VERSION)
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

        # ---- Episode ----
        self.max_steps = 1500

        # ---- Teams ----
        self.n_agents = 4
        self.team_A = [0, 1]
        self.team_B = [2, 3]

        # ---- Action: [nx, nz, bank angle, fire] ----
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0, -np.deg2rad(60), 0.0], dtype=np.float32),
            high=np.array([1.5, 3.0,  np.deg2rad(60), 1.0], dtype=np.float32),
        )

        # ---- Observation ----
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(23,), dtype=np.float32
        )

        # ---- Weapon / WEZ ----
        self.wez_R = 900.0
        self.wez_ang = np.deg2rad(25.0)
        self.lead_gate = np.deg2rad(20.0)
        self.bullet_speed = 300.0
        self.base_pk = 0.55

        # ---- Ammo / Fire ----
        self.initial_ammo = 80
        self.fire_cd_steps = 3

        # ---- Reward ----
        self.track_w = 0.02
        self.kill_w = 10.0

        # ---- Opponent ----
        self.heuristic_bot = HeuristicBot3DOF()

        # ---- Logging ----
        self.enable_logging = False
        self.log_dir = "runs/runs_2v2/replays"
        self.episode_tag = "ep"

        self.reset()

    # =====================================================
    # RESET
    # =====================================================
    def reset(self):
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

        self.hp = np.ones(self.n_agents)
        self.ammo = np.full(self.n_agents, self.initial_ammo)
        self.fire_cd = np.zeros(self.n_agents, dtype=int)
        self.wez_timer = np.zeros(len(self.team_A))

        self.fire_events = []
        self._traj = {"state": [], "action": [], "reward": [], "hp": [], "ammo": []}

        self.prev_enemy_dist = {
            i: {j: None for j in self.team_B} for i in self.team_A
        }

        return self._obs_all()

    # =====================================================
    # DYNAMICS
    # =====================================================
    def _integrate_3dof(self, i, nx, nz, mu):
        x, y, h, V, psi, gamma = self.s[i]

        dx = V * np.cos(psi) * np.cos(gamma)
        dy = V * np.sin(psi) * np.cos(gamma)
        dh = V * np.sin(gamma)

        dV = self.g * (nx - np.sin(gamma))
        dpsi = self.g * nz * np.sin(mu) / max(V * np.cos(gamma), 1e-3)
        dgamma = self.g * (nz * np.cos(mu) - np.cos(gamma)) / max(V, 1e-3)

        self.s[i, 0] += dx * self.dt
        self.s[i, 1] += dy * self.dt
        self.s[i, 2] += dh * self.dt
        self.s[i, 3] = np.clip(V + dV * self.dt, self.v_min, self.v_max)
        self.s[i, 4] = (psi + dpsi * self.dt + np.pi) % (2*np.pi) - np.pi
        self.s[i, 5] = np.clip(gamma + dgamma * self.dt,
                                -self.gamma_max, self.gamma_max)

    # =====================================================
    # GEOMETRY
    # =====================================================
    def _rel_geom(self, i, j):
        dx = self.s[j, 0] - self.s[i, 0]
        dy = self.s[j, 1] - self.s[i, 1]
        R = np.hypot(dx, dy)

        psi_i = self.s[i, 4]
        psi_j = self.s[j, 4]

        brg = (np.arctan2(dy, dx) - psi_i + np.pi) % (2*np.pi) - np.pi
        aoff = (psi_j - psi_i + np.pi) % (2*np.pi) - np.pi

        vij = self.s[j, 3] * np.array([np.cos(psi_j), np.sin(psi_j)])
        vii = self.s[i, 3] * np.array([np.cos(psi_i), np.sin(psi_i)])
        clos = -(dx*(vij[0]-vii[0]) + dy*(vij[1]-vii[1])) / (R+1e-6)

        return R, brg, aoff, clos

    def _lead_error(self, i, j):
        dx = self.s[j, 0] - self.s[i, 0]
        dy = self.s[j, 1] - self.s[i, 1]
        psi_i = self.s[i, 4]
        psi_j = self.s[j, 4]
        vj = self.s[j, 3]

        R = np.hypot(dx, dy)
        t = R / (self.bullet_speed + 1e-6)

        lead_x = dx + vj*np.cos(psi_j)*t
        lead_y = dy + vj*np.sin(psi_j)*t
        brg = (np.arctan2(lead_y, lead_x) - psi_i + np.pi) % (2*np.pi) - np.pi
        return abs(brg)

    # =====================================================
    # OBSERVATIONS
    # =====================================================
    def _obs_agent(self, i):
        obs = []
        V, psi, gamma = self.s[i, 3], self.s[i, 4], self.s[i, 5]

        obs.extend([
            (V-self.v_min)/(self.v_max-self.v_min),
            np.sin(psi), np.cos(psi),
            np.sin(gamma), np.cos(gamma),
        ])

        teammates = [k for k in self.team_A if k != i]
        enemies = self.team_B

        for j in teammates + enemies:
            R, brg, aoff, clos = self._rel_geom(i, j)
            obs.extend([
                np.clip(R/4000.0, 0, 1),
                np.sin(brg), np.cos(brg),
                np.sin(aoff), np.cos(aoff),
                np.tanh(clos/200.0)
            ])

        return np.clip(np.array(obs, dtype=np.float32), -1, 1)

    def _obs_all(self):
        return {i: self._obs_agent(i) for i in self.team_A}

    # =====================================================
    # FIRE LOGIC
    # =====================================================
    def _handle_fire(self, shooter, target, reward=None):
        if self.hp[shooter] <= 0 or self.hp[target] <= 0:
            return False
        if self.fire_cd[shooter] > 0 or self.ammo[shooter] <= 0:
            return False

        R, brg, aoff, _ = self._rel_geom(shooter, target)
        inside_wez = (R < self.wez_R and abs(brg) < self.wez_ang and abs(aoff) < self.wez_ang)
        if not inside_wez:
            return False

        self.ammo[shooter] -= 1
        self.fire_cd[shooter] = self.fire_cd_steps

        pk = self.base_pk * np.exp(-(aoff/self.lead_gate)**2)
        hit = self.rng.random() < np.clip(pk, 0.1, 0.9)

        if hit:
            self.hp[target] -= 1.0

        self.fire_events.append({
            "t": self.steps,
            "shooter": shooter,
            "target": target,
            "hit": hit,
            "shooter_pos": self.s[shooter,:2].copy(),
            "target_pos": self.s[target,:2].copy()
        })

        if reward is not None:
            reward[shooter] -= 0.002
            if hit:
                reward[shooter] += self.kill_w
                for tm in self.team_A:
                    if tm != shooter:
                        reward[tm] += 0.5

        return self.hp[target] <= 0

    # =====================================================
    # STEP
    # =====================================================
    def step(self, actions):
        self.steps += 1
        reward = {i: 0.0 for i in self.team_A}
        done, info = False, {}
        dists = []

        self.fire_cd = np.maximum(self.fire_cd-1, 0)

        # TEAM A
        for i in self.team_A:
            if self.hp[i] <= 0:
                continue
            nx, nz, mu, fire = actions[i]
            self._integrate_3dof(i, nx, nz, mu)
            if fire > 0.5:
                for j in self.team_B:
                    if self._handle_fire(i, j, reward):
                        done, info = True, {"winner":"A","termination":"kill"}

        # TEAM B
        for j in self.team_B:
            if self.hp[j] <= 0:
                continue
            obs_j = self._obs_agent(j)
            nx, nz, mu, fire = self.heuristic_bot.act(obs_j)
            self._integrate_3dof(j, nx, nz, mu)
            if fire > 0.5:
                for i in self.team_A:
                    if self._handle_fire(j, i):
                        done, info = True, {"winner":"B","termination":"kill"}

        # REWARD SHAPING
        for i in self.team_A:
            if self.hp[i] <= 0:
                continue
            tm = [k for k in self.team_A if k!=i][0]
            Rtm, _, _, _ = self._rel_geom(i, tm)
            reward[i] += 0.03*np.exp(-abs(Rtm - 1500)/800)

            for j in self.team_B:
                if self.hp[j] <= 0:
                    continue
                R, brg, _, _ = self._rel_geom(i,j)
                dists.append(R)
                reward[i] += 0.4*np.exp(-R/1500)
                prev = self.prev_enemy_dist[i][j]
                if prev is not None:
                    reward[i] += np.clip(0.003 * (prev-R), -0.05, 0.05)
                self.prev_enemy_dist[i][j]=R
                reward[i] += self.track_w*np.cos(self._lead_error(i, j))

        # TIME LIMIT
        if self.steps>=self.max_steps and not done:
            done = True
            info["termination"]="time_limit"
            hpA,hpB = self.hp[self.team_A].sum(),self.hp[self.team_B].sum()
            info["winner"] = "A" if hpA>hpB else "B" if hpB > hpA else "draw"

        if self.enable_logging:
            self._traj["state"].append(self.s.copy())
            self._traj["action"].append(actions.copy())
            self._traj["reward"].append(reward.copy())
            self._traj["hp"].append(self.hp.copy())
            self._traj["ammo"].append(self.ammo.copy())

        if done and self.enable_logging:
            os.makedirs(self.log_dir,exist_ok=True)
            np.savez_compressed(
                os.path.join(self.log_dir,f"episode_{self.episode_tag}_{info.get('winner','X')}.npz"),
                state=np.array(self._traj["state"]),
                action=np.array(self._traj["action"]),
                reward=np.array(self._traj["reward"]),
                hp=np.array(self._traj["hp"]),
                ammo=np.array(self._traj["ammo"]),
                fire_events=np.array(self.fire_events,dtype=object)
            )

        if dists:
            info["min_dist"] = float(np.min(dists))
            info["avg_dist"] = float(np.mean(dists))

        return self._obs_all(), reward, done, info
