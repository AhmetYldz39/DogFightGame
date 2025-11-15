# dogfight_env.py
from collections import deque
import numpy as np
import gymnasium as gym


class Dogfight1v1:

    def __init__(self, dt=0.1, arena=4000.0, seed=0):
        self.dt = dt
        self.rng = np.random.default_rng(seed)
        self.arena = arena
        self.max_steps = int(240 / self.dt)  # ~4 dk ep. (0.1 dt için 2400 adım)
        # Dinamik parametreler
        self.BDEL = np.deg2rad(6.0)
        self.vmin, self.vmax = 60.0, 240.0
        self.turn_k = 0.6     # bank->turn kazancı
        self.thr_k = 20.0     # throttle ivme kazancı

        # Curriculum-friendly WEZ ve PK (daha sonra sıkılaştıracağız)
        self.wez_R = 900.0
        self.wez_ang = np.deg2rad(30.0)
        self.base_pk = 0.65
        self.bullet_speed = 300.0
        self.lead_gate_tol = np.deg2rad(28.0)

        # Komut gecikmeleri
        self.delay_bank_steps = 2
        self.delay_thr_steps = 2
        self.delay_fire_steps = 0  # tetikte gecikme genelde istemiyoruz

        # --- Ölçüm (sensor) gürültüsü ---
        self.noise_bearing_std = np.deg2rad(1.0)
        self.noise_aoff_std = np.deg2rad(1.5)
        self.noise_range_std = 2.0  # metre
        self.noise_clos_std = 2.0   # m/s
        self.noise_speed_std = 0.5  # m/s
        self.noise_bank_std = np.deg2rad(0.5)

        # Süreç gürültüsü / rüzgâr (küçük)
        self.wind_vel_std = 0.5  # m/s

        # --- Gözlem boyutu (sin/cos açı kodlama) ---
        obs_dim = 10  # [R, sin/cos(brg), sin/cos(aoff), clos, v_norm, sin/cos(bank), ammo]
        if hasattr(self, "boost_energy_max"):
            obs_dim += 1  # opsiyonel boost_norm

        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )
        # Action space: continuous (bank_rate, throttle, trigger_prob)
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([+1.0, 1.0, 1.0], dtype=np.float32),
        )

        self.reset()

    def reset(self):
        self.steps = 0
        # [x, y, v, psi, bank] x 2
        self.s = np.zeros((2, 5), dtype=np.float32)

        # Curriculum spawn (kolay): saldıran (0) hedefe yakın ve uygun açıyla
        easy = True
        if easy:
            self.s[1, :] = [
                0.0,
                0.0,
                self.rng.uniform(140, 180),
                self.rng.uniform(-np.pi, np.pi),
                0.0,
            ]
            psi_t = self.s[1, 3]
            R0 = self.rng.uniform(600, 1000)  # daha yakın
            off = self.rng.uniform(-np.deg2rad(25), np.deg2rad(25))
            self.s[0, :] = [
                -R0 * np.cos(psi_t),
                -R0 * np.sin(psi_t),
                self.rng.uniform(150, 190),
                psi_t + off,
                0.0,
            ]
        else:
            # Genel rastgele spawn
            R0 = self.rng.uniform(1000, 1800)
            ang = self.rng.uniform(-np.pi / 2, np.pi / 2)
            self.s[0, :] = [0.0, 0.0, self.rng.uniform(120, 180), ang, 0.0]
            self.s[1, :] = [
                R0 * np.cos(ang + np.pi),
                R0 * np.sin(ang + np.pi),
                self.rng.uniform(120, 180),
                ang + np.pi,
                0.0,
            ]

        self.ammo = np.array([120, 120], dtype=np.int32)
        self.hp = np.array([1.0, 1.0], dtype=np.float32)

        # Komut gecikme tamponları
        neutral = (0.0, 0.5, 0.0)  # bank_rate=0, throttle=orta, trigger=0
        maxlen = max(self.delay_bank_steps, self.delay_thr_steps, self.delay_fire_steps) + 1
        self.cmd_buf = {
            0: deque([neutral] * maxlen, maxlen=maxlen),
            1: deque([neutral] * maxlen, maxlen=maxlen),
        }

        # jerk cezası için önceki bank kaydı
        self._prev_bank = np.zeros(2, dtype=np.float32)

        return self._obs_all()

    def step(self, actions):  # actions: {0:(bank_rate,thr_cmd,trigger_p), 1:(...)}, continuous
        self.steps += 1
        r = np.zeros(2, dtype=np.float32)
        r += -2e-5  # küçük zaman cezası (azaltıldı)
        done = False
        info = {}

        # 1) Gelen komutları gecikme tamponlarına yaz
        for i in (0, 1):
            self.cmd_buf[i].append(actions[i])

        # 2) Gecikmeli uygulanacak komutları oku
        delayed_actions = {}
        for i in (0, 1):
            a_latest = np.asarray(self.cmd_buf[i][-1], dtype=float).ravel()
            if a_latest.size < 3:
                a_latest = np.pad(a_latest, (0, 3 - a_latest.size))
            bank_rate, thr_cmd, trig = a_latest

            if self.delay_bank_steps > 0:
                a = np.asarray(self.cmd_buf[i][-(self.delay_bank_steps + 1)], dtype=float).ravel()
                if a.size >= 1:
                    bank_rate = float(a[0])
            if self.delay_thr_steps > 0:
                a = np.asarray(self.cmd_buf[i][-(self.delay_thr_steps + 1)], dtype=float).ravel()
                if a.size >= 2:
                    thr_cmd = float(a[1])
            if self.delay_fire_steps > 0:
                a = np.asarray(self.cmd_buf[i][-(self.delay_fire_steps + 1)], dtype=float).ravel()
                if a.size >= 3:
                    trig = float(a[2])

            bank_rate = float(np.clip(bank_rate, -1.0, 1.0))
            thr_cmd = float(np.clip(thr_cmd, 0.0, 1.0))
            trig = float(np.clip(trig, 0.0, 1.0))

            delayed_actions[i] = (bank_rate, thr_cmd, trig)

        # 3) Durum güncelle (continuous)
        bank_rate_max = getattr(self, "bank_rate_max", np.deg2rad(20.0))  # rad/s
        for i in (0, 1):
            bank_rate, thr_cmd, _ = delayed_actions[i]

            prev_bank = self.s[i, 4]
            self.s[i, 4] = np.clip(
                prev_bank + (bank_rate * bank_rate_max) * self.dt,
                -np.deg2rad(60),
                np.deg2rad(60),
            )
            # heading
            self.s[i, 3] = (self.s[i, 3] + self.turn_k * self.s[i, 4] * self.dt + np.pi) % (2 * np.pi) - np.pi
            # speed
            v_target = self.vmin + thr_cmd * (self.vmax - self.vmin)
            v = self.s[i, 2] + self.thr_k * (v_target - self.s[i, 2]) / self.vmax * self.dt
            self.s[i, 2] = np.clip(v, self.vmin, self.vmax)

            # jerk cezası (bank değişimi)
            r[i] += -0.0005 * abs(self.s[i, 4] - self._prev_bank[i])
            self._prev_bank[i] = self.s[i, 4]

        # 4) Konum + süreç gürültüsü
        for i in (0, 1):
            x, y, v, psi, _ = self.s[i]
            wx = self.rng.normal(0.0, self.wind_vel_std)
            wy = self.rng.normal(0.0, self.wind_vel_std)
            self.s[i, 0] = x + (v * np.cos(psi) + wx) * self.dt
            self.s[i, 1] = y + (v * np.sin(psi) + wy) * self.dt

        # 5) Arena sınırı cezası
        for i in (0, 1):
            if np.hypot(self.s[i, 0], self.s[i, 1]) > self.arena:
                r[i] -= 0.01

        # 6) Relatif metrik
        def rel(i, j):
            dx, dy = self.s[j, 0] - self.s[i, 0], self.s[j, 1] - self.s[i, 1]
            R = np.hypot(dx, dy)
            brg = (np.arctan2(dy, dx) - self.s[i, 3] + np.pi) % (2 * np.pi) - np.pi
            aoff = (self.s[i, 3] - self.s[j, 3] + np.pi) % (2 * np.pi) - np.pi
            vij = self.s[j, 2] * np.array([np.cos(self.s[j, 3]), np.sin(self.s[j, 3])])
            vii = self.s[i, 2] * np.array([np.cos(self.s[i, 3]), np.sin(self.s[i, 3])])
            relv = vij - vii
            clos = -(dx * relv[0] + dy * relv[1]) / (R + 1e-6)
            return R, brg, aoff, clos

        # 7) Ödül shaping + atış
        for i in (0, 1):
            j = 1 - i
            R, brg, aoff, clos = rel(i, j)

            # Lead-angle (ön nişan)
            dx, dy = self.s[j, 0] - self.s[i, 0], self.s[j, 1] - self.s[i, 1]
            psi_i = self.s[i, 3]
            psi_j = self.s[j, 3]
            v_j = self.s[j, 2]
            vjx, vjy = v_j * np.cos(psi_j), v_j * np.sin(psi_j)
            t_hit = R / (self.bullet_speed + 1e-6)
            lead_x = dx + vjx * t_hit
            lead_y = dy + vjy * t_hit
            lead_bearing = (np.arctan2(lead_y, lead_x) - psi_i + np.pi) % (2 * np.pi) - np.pi
            lead_err = abs(lead_bearing)

            inside_wez = (R < self.wez_R) and (abs(brg) < self.wez_ang)

            # Shaping (cömert)
            r[i] += 0.008 * inside_wez
            r[i] += -0.0004 * abs(aoff)
            r[i] += -0.00008 * R

            # Tetik
            _, _, trig = delayed_actions[i]
            fire = (trig > 0.5)

            if fire and self.ammo[i] > 0:
                self.ammo[i] -= 1
                if inside_wez:
                    # WEZ içi bonus (erken öğrenme)
                    r[i] += 0.01
                    # pk: lead hatasına bağlı
                    pk = self.base_pk * np.exp(- (lead_err / self.lead_gate_tol) ** 2)
                    pk = float(np.clip(pk, 0.10, 0.90))
                    if self.rng.random() < pk:
                        r[i] += 0.2
                        r[j] -= 0.2
                        self.hp[j] -= 0.5
                        if self.hp[j] <= 0.0:
                            r[i] += 0.8
                            r[j] -= 0.8
                            done = True
                            info["winner"] = i
                else:
                    # WEZ dışı ateşe yumuşak ceza (erken aşama)
                    r[i] -= 0.01

        # 8) Bitiş ve gözlem
        obs = self._obs_all()
        if self.steps >= self.max_steps:
            done = True

        # Güvenlik: NaN/Inf yakala (debug için)
        if not (np.all(np.isfinite(self.s)) and np.all(np.isfinite(obs[0])) and np.all(np.isfinite(obs[1]))):
            done = True
            info["truncated"] = True
            info["nan_guard"] = True

        info["hp0"] = float(self.hp[0])
        info["hp1"] = float(self.hp[1])
        return obs, r, done, info

    def _obs(self, i):
        """Ajan i için GÜRÜLTÜLÜ (ölçüm) gözlem vektörü (sin/cos açı kodlaması)."""
        j = 1 - i

        # Gerçek relatifler
        dx, dy = self.s[j, 0] - self.s[i, 0], self.s[j, 1] - self.s[i, 1]
        R = np.hypot(dx, dy)
        brg = (np.arctan2(dy, dx) - self.s[i, 3] + np.pi) % (2 * np.pi) - np.pi
        aoff = (self.s[i, 3] - self.s[j, 3] + np.pi) % (2 * np.pi) - np.pi
        vij = self.s[j, 2] * np.array([np.cos(self.s[j, 3]), np.sin(self.s[j, 3])])
        vii = self.s[i, 2] * np.array([np.cos(self.s[i, 3]), np.sin(self.s[i, 3])])
        relv = vij - vii
        clos = -(dx * relv[0] + dy * relv[1]) / (R + 1e-6)

        # Ölçüm gürültüsü
        R_n = R + self.rng.normal(0.0, self.noise_range_std)
        brg_n = brg + self.rng.normal(0.0, self.noise_bearing_std)
        aoff_n = aoff + self.rng.normal(0.0, self.noise_aoff_std)
        clos_n = clos + self.rng.normal(0.0, self.noise_clos_std)
        v_n = self.s[i, 2] + self.rng.normal(0.0, self.noise_speed_std)
        bank_n = self.s[i, 4] + self.rng.normal(0.0, self.noise_bank_std)

        # Sin/cos açı kodlama
        brg_s, brg_c = np.sin(brg_n), np.cos(brg_n)
        aoff_s, aoff_c = np.sin(aoff_n), np.cos(aoff_n)
        bank_s, bank_c = np.sin(bank_n), np.cos(bank_n)

        ammo_cap = 120.0
        o_list = [
            np.clip(R_n / 2000.0, 0.0, 1.0),
            brg_s, brg_c,
            aoff_s, aoff_c,
            np.tanh(clos_n / 200.0),
            (v_n - self.vmin) / (self.vmax - self.vmin),
            bank_s, bank_c,
            min(1.0, self.ammo[i] / ammo_cap),
        ]
        if hasattr(self, "boost_energy") and hasattr(self, "boost_energy_max"):
            o_list.append(float(self.boost_energy[i] / (self.boost_energy_max + 1e-6)))

        o = np.array(o_list, dtype=np.float32)
        return np.clip(o, -1.0, 1.0)

    def _obs_all(self):
        return {0: self._obs(0), 1: self._obs(1)}
