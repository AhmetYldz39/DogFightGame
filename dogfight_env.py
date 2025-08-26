# dogfight_env.py
from collections import deque
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
        self.thr_k = 20.0    # throttle ivme kazancı
        self.wez_R = 600.0
        self.wez_ang = np.deg2rad(20.0)

        # bank ve throttle komutları kaç frame gecikmeli uygulansın?
        self.delay_bank_steps = 2  # 0 -> gecikme yok
        self.delay_thr_steps = 2
        self.delay_fire_steps = 0  # tetikte gecikme istemiyorsan 0 bırak

        # ==== YENİ: ölçüm (sensor) gürültüsü ====
        # Açı std'leri radyan cinsinden:
        self.noise_bearing_std = np.deg2rad(1.0)  # 1°
        self.noise_aoff_std = np.deg2rad(1.5)  # 1.5°
        # Diğerleri doğal birimlerinde:
        self.noise_range_std = 2.0  # metre
        self.noise_clos_std = 2.0  # m/s
        self.noise_speed_std = 0.5  # m/s
        self.noise_bank_std = np.deg2rad(0.5)  # 0.5°

        # çok küçük süreç gürültüsü / rüzgâr
        self.wind_vel_std = 0.5  # m/s ~ her adımda rastgele sürüklenme

        self.reset()

    def reset(self):
        self.steps = 0
        # [x, y, v, psi, bank] - for two agent
        self.s = np.zeros((2, 5), dtype=np.float32)
        # random spawn: karşılıklı, hafif açılı
        R0 = self.rng.uniform(1000, 1800)
        ang = self.rng.uniform(-np.pi/2, np.pi/2)
        self.s[0, :] = [0.0, 0.0, self.rng.uniform(120, 180), ang, 0.0]
        self.s[1, :] = [R0*np.cos(ang+np.pi), R0*np.sin(ang+np.pi),
                       self.rng.uniform(120, 180), ang+np.pi, 0.0]

        easy = False  # <— eğitimde True, sonra False yap
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
        self.ammo = np.array([120, 120], dtype=np.int32)
        self.hp = np.array([1.0, 1.0], dtype=np.float32)

        # ==== YENİ: gecikme tamponları ====
        # deque(maxlen) ile sabit uzunluklu kuyruk: en eski eleman uygulanan komut
        neutral = (1, 1, 0)  # bank=straight, throttle=hold, fire=0
        self.cmd_buf = {
            0: deque([neutral] * (max(self.delay_bank_steps, self.delay_thr_steps, self.delay_fire_steps) + 1),
                     maxlen=max(self.delay_bank_steps, self.delay_thr_steps, self.delay_fire_steps) + 1),
            1: deque([neutral] * (max(self.delay_bank_steps, self.delay_thr_steps, self.delay_fire_steps) + 1),
                     maxlen=max(self.delay_bank_steps, self.delay_thr_steps, self.delay_fire_steps) + 1),
        }

        obs = self._obs_all()
        return obs

    def step(self, actions):  # actions: {0:(bank_cmd,thr_cmd,fire), 1:(...)}
        self.steps += 1
        r = np.zeros(2, dtype=np.float32)
        done = False
        info = {}

        # ---- 1) Gelen komutları gecikme tamponlarına yaz ----
        for i in (0, 1):
            self.cmd_buf[i].append(actions[i])

        # ---- 2) Gecikmeli uygulanacak komutları oku ----
        delayed_actions = {}
        for i in (0, 1):
            # deque[-1] en yeni, deque[0] en eski
            bank_cmd, thr_cmd, fire_cmd = self.cmd_buf[i][-1]  # varsayılan
            if getattr(self, "delay_bank_steps", 0) > 0:
                bank_cmd = self.cmd_buf[i][-(self.delay_bank_steps + 1)][0]
            if getattr(self, "delay_thr_steps", 0) > 0:
                thr_cmd = self.cmd_buf[i][-(self.delay_thr_steps + 1)][1]
            if getattr(self, "delay_fire_steps", 0) > 0:
                fire_cmd = self.cmd_buf[i][-(self.delay_fire_steps + 1)][2]
            delayed_actions[i] = (int(bank_cmd), int(thr_cmd), int(fire_cmd))

        # ---- 3) Durum güncelle (bank/throttle) ----
        for i in (0, 1):
            bank_cmd, thr_cmd, _ = delayed_actions[i]

            # bank update (−1,0,+1)*BDEL)
            self.s[i, 4] = np.clip(
                self.s[i, 4] + (bank_cmd - 1) * self.BDEL,
                -np.deg2rad(60), np.deg2rad(60)
            )
            # heading
            self.s[i, 3] = (self.s[i, 3] + self.turn_k * self.s[i, 4] * self.dt + np.pi) % (2 * np.pi) - np.pi
            # speed
            v = self.s[i, 2]
            v_target = self.vmax if thr_cmd == 2 else (self.vmin if thr_cmd == 0 else v)
            v = v + self.thr_k * (v_target - v) / self.vmax * self.dt
            self.s[i, 2] = np.clip(v, self.vmin, self.vmax)

        # ---- 4) Konum + süreç gürültüsü (küçük rüzgâr) ----
        for i in (0, 1):
            x, y, v, psi, _ = self.s[i]
            wx = self.rng.normal(0.0, getattr(self, "wind_vel_std", 0.0))
            wy = self.rng.normal(0.0, getattr(self, "wind_vel_std", 0.0))
            self.s[i, 0] = x + (v * np.cos(psi) + wx) * self.dt
            self.s[i, 1] = y + (v * np.sin(psi) + wy) * self.dt

        # ---- 5) Arena sınırı hafif ceza ----
        for i in (0, 1):
            if np.hypot(self.s[i, 0], self.s[i, 1]) > self.arena:
                r[i] -= 0.01

        # ---- 6) Relatif metrik yardımcı fonksiyonu ----
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

        # ---- 7) Ödül shaping + atış/kill mantığı ----
        for i in (0, 1):
            j = 1 - i
            R, brg, aoff, clos = rel(i, j)

            inside_wez = (R < self.wez_R) and (abs(brg) < self.wez_ang)
            # Shaping (güçlendirilmiş)
            r[i] += 0.004 * inside_wez
            r[i] += -0.0010 * abs(aoff)
            r[i] += -0.0002 * R

            # Gecikmiş fire komutu
            if delayed_actions[i][2] == 1 and self.ammo[i] > 0:
                self.ammo[i] -= 1
                if inside_wez:
                    # Biraz cömert pk (curriculum aşaması); kademeli normale döndürülebilir
                    pk = max(0.2, 0.7 - 0.02 * R / 100 - 0.4 * abs(aoff) / np.deg2rad(60))
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
                    # israf cezası (curriculum’da yumuşak; sonra artır)
                    r[i] -= 0.005

        # ---- 8) Episode bitişi / gözlem / info ----
        obs = self._obs_all()
        if self.steps >= self.max_steps:
            done = True

        info["hp0"] = float(self.hp[0])
        info["hp1"] = float(self.hp[1])
        return obs, r, done, info

    def _obs(self, i):
        """Ajan i için GÜRÜLTÜLÜ (ölçüm) gözlem vektörü döndürür."""
        j = 1 - i

        # --- gerçek (gürültüsüz) relatifler ---
        dx, dy = self.s[j, 0] - self.s[i, 0], self.s[j, 1] - self.s[i, 1]
        R = np.hypot(dx, dy)
        brg = (np.arctan2(dy, dx) - self.s[i, 3] + np.pi) % (2 * np.pi) - np.pi
        aoff = (self.s[i, 3] - self.s[j, 3] + np.pi) % (2 * np.pi) - np.pi
        vij = self.s[j, 2] * np.array([np.cos(self.s[j, 3]), np.sin(self.s[j, 3])])
        vii = self.s[i, 2] * np.array([np.cos(self.s[i, 3]), np.sin(self.s[i, 3])])
        relv = vij - vii
        clos = -(dx * relv[0] + dy * relv[1]) / (R + 1e-6)

        # --- ölçüm gürültüsü (normalize ETMEDEN önce ekle) ---
        R_n = R + self.rng.normal(0.0, getattr(self, "noise_range_std", 0.0))
        brg_n = brg + self.rng.normal(0.0, getattr(self, "noise_bearing_std", 0.0))
        aoff_n = aoff + self.rng.normal(0.0, getattr(self, "noise_aoff_std", 0.0))
        clos_n = clos + self.rng.normal(0.0, getattr(self, "noise_clos_std", 0.0))
        v_n = self.s[i, 2] + self.rng.normal(0.0, getattr(self, "noise_speed_std", 0.0))
        bank_n = self.s[i, 4] + self.rng.normal(0.0, getattr(self, "noise_bank_std", 0.0))

        # --- normalize et ---
        # NOT: ammo normalizasyonu için 120 kullandım; başlangıç cephanen farklıysa bu sabiti değiştir.
        o = np.array([
            np.clip(R_n / 2000.0, 0.0, 1.0),
            (((brg_n + np.pi) % (2 * np.pi)) - np.pi) / np.pi,
            (((aoff_n + np.pi) % (2 * np.pi)) - np.pi) / np.pi,
            np.tanh(clos_n / 200.0),
            (v_n - self.vmin) / (self.vmax - self.vmin),
            bank_n / np.deg2rad(60),
            min(1.0, self.ammo[i] / 120.0),  # <-- ammo cap 120 ise 1.0'a kadar çıkar
        ], dtype=np.float32)

        return np.clip(o, -1.0, 1.0)

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
