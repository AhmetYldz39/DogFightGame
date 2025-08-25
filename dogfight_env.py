# dogfight_env.py
import numpy as np


class Dogfight1v1:

    def __init__(self, dt=0.1, arena=4000.0, seed=0):
        self.dt = dt
        self.rng = np.random.default_rng(seed)
        self.arena = arena
        self.max_steps = int(240/self.dt)  # 3 dk ep.
        self.BDEL = np.deg2rad(6.0)
        self.vmin, self.vmax = 60.0, 240.0
        self.turn_k = 0.6     # bank->turn kazancı
        self.thr_k  = 20.0    # throttle ivme kazancı
        self.wez_R  = 600.0
        self.wez_ang= np.deg2rad(20.0)
        self.reset()

    def reset(self):
        self.steps = 0
        # [x, y, v, psi, bank] - for two agent
        self.s = np.zeros((2, 5), dtype=np.float32)
        # random spawn: karşılıklı, hafif açılı
        R0 = self.rng.uniform(800, 1400)
        ang = self.rng.uniform(-np.pi/6, np.pi/6)
        self.s[0, :] = [0.0, 0.0, self.rng.uniform(120, 180), ang, 0.0]
        self.s[1, :] = [R0*np.cos(ang+np.pi), R0*np.sin(ang+np.pi),
                       self.rng.uniform(120, 180), ang+np.pi, 0.0]

        easy = True  # <— eğitimde True, sonra False yap
        if easy:
            # hedefi 1, avcıyı 0 kabul edelim
            self.s[1, :] = [0.0, 0.0, self.rng.uniform(140, 180), self.rng.uniform(-np.pi, np.pi), 0.0]
            psi_t = self.s[1, 3]
            R0 = self.rng.uniform(600, 1000)  # daha yakın
            off = self.rng.uniform(-np.deg2rad(25), np.deg2rad(25))  # az açı hatası
            self.s[0, :] = [
                -R0 * np.cos(psi_t), -R0 * np.sin(psi_t),
                self.rng.uniform(150, 190),
                psi_t + off,
                0.0
            ]

        self.ammo = np.array([120, 120],dtype=np.int32)
        self.hp = np.array([1.0, 1.0], dtype=np.float32)
        obs = self._obs_all()
        return obs

    def step(self, actions):  # actions: {0:(bank_cmd,thr_cmd,fire), 1:(...)}
        self.steps += 1
        r = np.zeros(2, dtype=np.float32)
        done = False
        info = {}

        for i in (0,1):
            bank_cmd, thr_cmd, fire = actions[i]
            # bank update
            self.s[i, 4] = np.clip(self.s[i,4] + (bank_cmd-1)*self.BDEL, -np.deg2rad(60), np.deg2rad(60))
            # heading
            self.s[i, 3] = (self.s[i,3] + self.turn_k*self.s[i,4]*self.dt + np.pi)%(2*np.pi)-np.pi
            # speed
            v = self.s[i, 2]
            v_target = self.vmax if thr_cmd==2 else (self.vmin if thr_cmd==0 else v)
            v = v + self.thr_k * (v_target - v)/self.vmax * self.dt
            self.s[i, 2] = np.clip(v, self.vmin, self.vmax)

        # --- konum
        for i in (0, 1):
            x, y, v, psi, _ = self.s[i]
            self.s[i, 0] = x + v*np.cos(psi)*self.dt
            self.s[i, 1] = y + v*np.sin(psi)*self.dt

        # --- sınırlar (duvara hafif ceza)
        for i in (0, 1):
            if np.hypot(self.s[i, 0], self.s[i, 1]) > self.arena:
                r[i] -= 0.01

        # --- çatışma ölçümleri
        def rel(i, j):
            dx, dy = self.s[j, 0]-self.s[i, 0], self.s[j, 1]-self.s[i, 1]
            R = np.hypot(dx,dy)
            brg = (np.arctan2(dy, dx) - self.s[i, 3] + np.pi) % (2*np.pi) - np.pi
            # angle-off: hedefin kursuna göre senin bearing’in
            aoff = (self.s[i, 3] - self.s[j, 3] + np.pi) % (2*np.pi) - np.pi
            # closure
            vij = self.s[j, 2]*np.array([np.cos(self.s[j, 3]), np.sin(self.s[j, 3])])
            vii = self.s[i, 2]*np.array([np.cos(self.s[i, 3]), np.sin(self.s[i, 3])])
            relv = vij - vii
            clos = -(dx*relv[0] + dy*relv[1]) / (R+1e-6)
            return R, brg, aoff, clos

        # --- ödül shaping
        for i in (0, 1):
            j = 1 - i
            R, brg, aoff, clos = rel(i, j)
            inside_wez = (R < self.wez_R) and (abs(brg) < self.wez_ang)
            r[i] += 0.004 * inside_wez
            r[i] += -0.0005 * abs(aoff)
            r[i] += -0.0002 * R

            # atış
            if actions[i][2] == 1 and self.ammo[i] > 0:
                self.ammo[i] -= 1
                if inside_wez:
                    # basit pk modeli
                    pk = max(0.2, 0.7 - 0.02*R/100 - 0.4*abs(aoff)/np.deg2rad(60))
                    if self.rng.random() < pk:
                        # hit
                        r[i] += 0.2; r[j] -= 0.2
                        self.hp[j] -= 0.5
                        if self.hp[j] <= 0.0:
                            r[i] += 0.8
                            r[j] -= 0.8
                            done = True
                            info["winner"] = i
                else:
                    r[i] -= 0.005  # israf

        obs = self._obs_all()
        if self.steps >= self.max_steps:
            done = True

        info["hp0"] = float(self.hp[0])
        info["hp1"] = float(self.hp[1])

        return obs, r, done, info

    def _obs(self, i):
        j = 1-i
        R, brg, aoff, clos = self._rel_cached(i, j)
        # normalize
        o = np.array([
            np.clip(R/2000.0, 0, 1),
            brg/np.pi,
            aoff/np.pi,
            np.tanh(clos/200.0),
            (self.s[i, 2]-self.vmin)/(self.vmax-self.vmin),
            self.s[i, 4]/np.deg2rad(60),
            min(1.0, self.ammo[i]/60.0),
        ], dtype=np.float32)
        return o

    def _rel_cached(self, i, j):
        dx, dy = self.s[j, 0]-self.s[i, 0], self.s[j, 1]-self.s[i, 1]
        R = np.hypot(dx, dy)
        brg = (np.arctan2(dy, dx) - self.s[i, 3] + np.pi) % (2*np.pi) - np.pi
        aoff = (self.s[i, 3] - self.s[j, 3] + np.pi) % (2*np.pi)-np.pi
        vij = self.s[j, 2]*np.array([np.cos(self.s[j, 3]), np.sin(self.s[j, 3])])
        vii = self.s[i, 2]*np.array([np.cos(self.s[i, 3]), np.sin(self.s[i, 3])])
        relv = vij - vii
        clos = -(dx*relv[0] + dy*relv[1]) / (R + 1e-6)
        return R, brg, aoff, clos

    def _obs_all(self):
        return {0: self._obs(0), 1: self._obs(1)}
