# dogfight_env.py
from collections import deque
import numpy as np
import gymnasium as gym


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
        self.wez_R = 800.0
        self.wez_ang = np.deg2rad(25.0)
        self.base_pk = 0.65
        self.bullet_speed = 300.0

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

        # --- gözlem boyutu ayarı ---
        # Eğer boost sistemi yoksa 10, varsa 11 elemanlı.
        obs_dim = 10
        if hasattr(self, "boost_energy_max"):
            obs_dim += 1

        self.observation_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(obs_dim,),
            dtype=np.float32
        )

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

    def step(self, actions):  # actions: {0:(bank_rate,thr_cmd,trigger_p), 1:(...)}
        self.steps += 1
        r = np.zeros(2, dtype=np.float32)
        r += -1e-4  # küçük zaman cezası
        done = False
        info = {}

        # ---- 1) Gelen komutları gecikme tamponlarına yaz ----
        for i in (0, 1):
            self.cmd_buf[i].append(actions[i])

        # ---- 2) Gecikmeli uygulanacak komutları oku ----
        delayed_actions = {}
        for i in (0, 1):
            # deque[-1] en yeni, deque[0] en eski
            a_latest = np.asarray(self.cmd_buf[i][-1], dtype=float).ravel()
            # güvenli: yoksa n=3'e doldur
            if a_latest.size < 3:
                pad = np.zeros(3 - a_latest.size, dtype=float)
                a_latest = np.concatenate([a_latest, pad])

            bank_rate, thr_cmd, trig = a_latest  # defaults (gecikmesiz)
            if getattr(self, "delay_bank_steps", 0) > 0:
                a = np.asarray(self.cmd_buf[i][-(self.delay_bank_steps + 1)], dtype=float).ravel()
                if a.size >= 1:
                    bank_rate = float(a[0])
            if getattr(self, "delay_thr_steps", 0) > 0:
                a = np.asarray(self.cmd_buf[i][-(self.delay_thr_steps + 1)], dtype=float).ravel()
                if a.size >= 2:
                    thr_cmd = float(a[1])
            if getattr(self, "delay_fire_steps", 0) > 0:
                a = np.asarray(self.cmd_buf[i][-(self.delay_fire_steps + 1)], dtype=float).ravel()
                if a.size >= 3:
                    trig = float(a[2])

            # clip güvenliği
            bank_rate = float(np.clip(bank_rate, -1.0, 1.0))
            thr_cmd = float(np.clip(thr_cmd, 0.0, 1.0))
            trig = float(np.clip(trig, 0.0, 1.0))

            delayed_actions[i] = (bank_rate, thr_cmd, trig)

        # ---- 3) Durum güncelle (bank/throttle continuous) ----
        bank_rate_max = getattr(self, "bank_rate_max", np.deg2rad(20.0))  # rad/s
        for i in (0, 1):
            bank_rate, thr_cmd, _ = delayed_actions[i]

            # bank açısı (rad) – sürekli hızla değişim
            prev_bank = self.s[i, 4]
            self.s[i, 4] = np.clip(
                prev_bank + (bank_rate * bank_rate_max) * self.dt,
                -np.deg2rad(60), np.deg2rad(60)
            )
            # heading
            self.s[i, 3] = (self.s[i, 3] + self.turn_k * self.s[i, 4] * self.dt + np.pi) % (2 * np.pi) - np.pi
            # hız: hedef hız vmin..vmax arasında linear interpolation
            v_target = self.vmin + thr_cmd * (self.vmax - self.vmin)
            v = self.s[i, 2] + self.thr_k * (v_target - self.s[i, 2]) / self.vmax * self.dt
            self.s[i, 2] = np.clip(v, self.vmin, self.vmax)

            # komut yumuşaklığı (bank değişimine küçük ceza)
            if not hasattr(self, "_prev_bank"):
                self._prev_bank = np.zeros(2, dtype=np.float32)
            r[i] += -0.0005 * abs(self.s[i, 4] - self._prev_bank[i])
            self._prev_bank[i] = self.s[i, 4]

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

            # --- Lead-angle (ön nişan) hesabı ---
            dx, dy = self.s[j, 0] - self.s[i, 0], self.s[j, 1] - self.s[i, 1]
            psi_i = self.s[i, 3]
            psi_j = self.s[j, 3]
            v_i, v_j = self.s[i, 2], self.s[j, 2]

            # Hedefin yönündeki hız bileşenleri
            vjx = v_j * np.cos(psi_j)
            vjy = v_j * np.sin(psi_j)

            # Hedefe ulaşma süresi yaklaşık R / bullet_speed
            t_hit = R / (self.bullet_speed + 1e-6)

            # Hedefin o sürede ilerleyeceği yer
            lead_x = dx + vjx * t_hit
            lead_y = dy + vjy * t_hit

            # Ajanın yönüne göre “lead bearing” açısı
            lead_bearing = (np.arctan2(lead_y, lead_x) - psi_i + np.pi) % (2 * np.pi) - np.pi

            # Hata: hedefin gelecekteki yönüyle ajanın nişan yönü arasındaki fark
            lead_err = abs(lead_bearing)

            inside_wez = (R < self.wez_R) and (abs(brg) < self.wez_ang)
            # Shaping
            r[i] += 0.004 * inside_wez
            r[i] += -0.0010 * abs(aoff)
            r[i] += -0.0001 * R

            # Continuous tetik: olasılık > 0.5 ise ateş say
            _, _, trig = delayed_actions[i]
            fire = (trig > 0.5)

            if fire and self.ammo[i] > 0:
                self.ammo[i] -= 1
                if inside_wez:
                    r[i] += 0.002  # gate/WEZ içi mikro bonus
                    pk = self.base_pk * np.exp(- (lead_err / np.deg2rad(20)) ** 2)
                    pk = float(np.clip(pk, 0.05, 0.85))
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
                    # WEZ dışı akılsız ateşe sert ceza
                    r[i] -= 0.01

        # ---- 8) Episode bitişi / gözlem / info ----
        obs = self._obs_all()
        if self.steps >= self.max_steps:
            done = True

        info["hp0"] = float(self.hp[0])
        info["hp1"] = float(self.hp[1])
        return obs, r, done, info

    def _obs(self, i):
        """Ajan i için GÜRÜLTÜLÜ (ölçüm) gözlem vektörü döndürür (sin/cos açı kodlamasıyla)."""
        j = 1 - i

        # --- gerçek (gürültüsüz) relatifler ---
        dx, dy = self.s[j, 0] - self.s[i, 0], self.s[j, 1] - self.s[i, 1]
        R = np.hypot(dx, dy)
        brg = (np.arctan2(dy, dx) - self.s[i, 3] + np.pi) % (2*np.pi) - np.pi
        aoff = (self.s[i, 3] - self.s[j, 3] + np.pi) % (2*np.pi) - np.pi
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

        # --- açıları sin/cos ile kodla ---
        brg_s, brg_c = np.sin(brg_n),  np.cos(brg_n)
        aoff_s, aoff_c = np.sin(aoff_n), np.cos(aoff_n)
        bank_s, bank_c = np.sin(bank_n), np.cos(bank_n)

        # --- normalize et ---
        ammo_cap = 120.0  # başlangıç cephanen farklıysa bunu değiştir
        o_list = [
            np.clip(R_n / 2000.0, 0.0, 1.0),      # mesafe [0,1] (~0..2000 m)
            brg_s, brg_c,                          # bearing sin/cos
            aoff_s, aoff_c,                        # angle-off sin/cos
            np.tanh(clos_n / 200.0),               # closure ~[-1,1]
            (v_n - self.vmin) / (self.vmax - self.vmin),  # hız [0,1]
            bank_s, bank_c,                        # bank sin/cos
            min(1.0, self.ammo[i] / ammo_cap),     # ammo [0,1]
        ]

        # boost enerjisi varsa ekle (opsiyonel)
        if hasattr(self, "boost_energy") and hasattr(self, "boost_energy_max"):
            o_list.append(float(self.boost_energy[i] / (self.boost_energy_max + 1e-6)))

        o = np.array(o_list, dtype=np.float32)

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
