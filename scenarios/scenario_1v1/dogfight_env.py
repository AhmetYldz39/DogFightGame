import numpy as np
from gymnasium import spaces


class Dogfight1v1:
    def __init__(self, dt=0.1, arena=4000.0, seed=0, timeout_as_loss=False):
        self.dt = dt
        self.arena = arena
        self.rng = np.random.default_rng(seed)

        # ---- Model Parameters ----
        self.g = 9.81
        self.mass = 1000.0
        self.thr_min = 0.0
        self.thr_max = 1.0
        self.bank_max = np.deg2rad(60.0)
        self.v_min = 100.0
        self.v_max = 300.0
        self.v_drag = 0.002

        # ---- WEZ / PK ----
        self.wez_R = 750.0
        self.wez_ang = np.deg2rad(22.0)
        self.base_pk = 0.55
        self.bullet_speed = 300.0
        self.lead_gate_tol = np.deg2rad(20.0)

        # ---- Reward weights ----
        self.track_w = 0.02    # cos(lead_err) shaping
        self.shot_w = 0.001    # good shot reward
        self.bad_shot_w = 0.01 # bad shot charge
        self.ammo_w = 0.01     # ammo dependency charge

        # ---- Ammo ----
        self.initial_ammo = 80

        # ---- Episode Limits ----
        self.timeout_as_loss = timeout_as_loss
        self.max_steps = 1200  # dt=0.1 ise ~120 saniye

        # ---- Fire cooldown ----
        self.fire_cd_steps = 3  # 3 step ateş edemez

        # ---- Action delays ----
        self.delay_bank_steps = 2
        self.delay_thr_steps = 2
        self.delay_fire_steps = 0

        # ---- OBS / ACTION SPACE TANIMI ----
        # Obs vektörünün boyutu: _obs() fonksiyonunda return ettiğin uzunluk
        # [R_norm, sin(brg), cos(brg), sin(aoff), cos(aoff),
        #  clos_norm, v_norm, sin(bank), cos(bank), ammo_norm]
        obs_dim = 10

        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        # Aksiyon: [bank_rate, throttle, trigger_prob]
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([+1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self.reset()

    # ------------------------------------------------------------

    def reset(self):
        self.s = np.zeros((2, 5), dtype=np.float32)
        for i in (0, 1):
            ang = self.rng.uniform(0, 2 * np.pi)
            rad = self.rng.uniform(1000, 1800)
            self.s[i, 0] = rad * np.cos(ang)
            self.s[i, 1] = rad * np.sin(ang)
            self.s[i, 2] = self.rng.uniform(170, 220)
            self.s[i, 3] = self.rng.uniform(-np.pi, np.pi)
            self.s[i, 4] = 0.0

        self.hp = np.ones(2, dtype=np.float32)
        self.ammo = np.array([self.initial_ammo, self.initial_ammo], dtype=np.int32)

        self.bank_queue = [[], []]
        self.thr_queue = [[], []]
        self.fire_queue = [[], []]

        self._prev_bank = np.zeros(2, dtype=np.float32)

        self.fire_cd = np.zeros(2, dtype=np.int32)

        # episode step sayacı
        self.steps = 0

        return self._obs_all()

    # ------------------------------------------------------------

    def _obs(self, i: int) -> np.ndarray:
        """
        Ajan i için GÜRÜLTÜLÜ (ölçüm) gözlem vektörü döndür.
        Obs vektörü boyutu: 10
          0: R_norm           (0..1,  0 -> 0m, 1 -> 2000m+)
          1: sin(brg)         (hedef bearing)
          2: cos(brg)
          3: sin(aoff)        (aim-off açısı)
          4: cos(aoff)
          5: tanh(clos/200)   (closure, +-200 m/s skala)
          6: v_norm           ((v - vmin)/(vmax - vmin))
          7: sin(bank)
          8: cos(bank)
          9: ammo_norm        (ammo / ammo_cap, 0..1)
        """
        j = 1 - i

        # --- gerçek (gürültüsüz) relatifler ---
        dx = self.s[j, 0] - self.s[i, 0]
        dy = self.s[j, 1] - self.s[i, 1]
        R = np.hypot(dx, dy)

        own_hdg = self.s[i, 3]
        tgt_hdg = self.s[j, 3]

        # bearing -pi..pi
        brg = (np.arctan2(dy, dx) - own_hdg + np.pi) % (2 * np.pi) - np.pi
        # aim-off -pi..pi
        aoff = (tgt_hdg - own_hdg + np.pi) % (2 * np.pi) - np.pi

        vij = self.s[j, 2] * np.array([np.cos(tgt_hdg), np.sin(tgt_hdg)])
        vii = self.s[i, 2] * np.array([np.cos(own_hdg), np.sin(own_hdg)])
        vrel = vij - vii
        clos = -(dx * vrel[0] + dy * vrel[1]) / (R + 1e-6)

        v_true = self.s[i, 2]
        bank_true = self.s[i, 4]

        # --- ölçüm gürültüsü EKLE (normalize ETMEDEN önce) ---
        R_n = R + self.rng.normal(loc=0.0, scale=getattr(self, "noise_range_std", 0.0))
        brg_n = brg + self.rng.normal(loc=0.0, scale=getattr(self, "noise_bearing_std", 0.0))
        aoff_n = aoff + self.rng.normal(loc=0.0, scale=getattr(self, "noise_aoff_std", 0.0))
        clos_n = clos + self.rng.normal(loc=0.0, scale=getattr(self, "noise_clos_std", 0.0))
        v_n = v_true + self.rng.normal(loc=0.0, scale=getattr(self, "noise_speed_std", 0.0))
        bank_n = bank_true + self.rng.normal(loc=0.0, scale=getattr(self, "noise_bank_std", 0.0))

        o = np.array(
            [
                np.clip(R_n / 2000.0, 0.0, 1.0),
                np.sin(brg_n),
                np.cos(brg_n),
                np.sin(aoff_n),
                np.cos(aoff_n),
                np.tanh(clos_n / 200.0),
                (v_n - self.v_min) / (self.v_max - self.v_min + 1e-6),
                np.sin(bank_n),
                np.cos(bank_n),
                min(1.0, self.ammo[i] / self.initial_ammo),
            ],
            dtype=np.float32,
        )

        return np.clip(o, -1.0, 1.0)

    def _obs_all(self):
        return [self._obs(0), self._obs(1)]

    # ------------------------------------------------------------

    def step(self, actions):
        done = False
        info = {}
        r = np.zeros(2, dtype=np.float32)

        # step sayacını artır
        self.steps += 1

        # ---- Utility: relative geometry ----
        def rel(i, j):
            dx = self.s[j, 0] - self.s[i, 0]
            dy = self.s[j, 1] - self.s[i, 1]
            R = np.hypot(dx, dy)
            psi_i = self.s[i, 3]
            psi_j = self.s[j, 3]
            brg = (np.arctan2(dy, dx) - psi_i + np.pi) % (2 * np.pi) - np.pi
            aoff = ((psi_j - psi_i + np.pi) % (2 * np.pi)) - np.pi
            vij = self.s[j, 2] * np.cos(psi_j - psi_i) - self.s[i, 2]
            clos = vij
            return R, brg, aoff, clos

        # ----------------------------------------------------
        # 1. ACTION PROCESSING WITH DELAYS
        # ----------------------------------------------------
        delayed_actions = []
        for i in (0, 1):
            a_bank = float(np.clip(actions[i][0], -1.0, 1.0))
            a_thr = float(np.clip(actions[i][1], 0.0, 1.0))
            a_fire = float(np.clip(actions[i][2], 0.0, 1.0))

            self.bank_queue[i].append(a_bank)
            if len(self.bank_queue[i]) > self.delay_bank_steps:
                a_bank = self.bank_queue[i].pop(0)

            self.thr_queue[i].append(a_thr)
            if len(self.thr_queue[i]) > self.delay_thr_steps:
                a_thr = self.thr_queue[i].pop(0)

            self.fire_queue[i].append(a_fire)
            if len(self.fire_queue[i]) > self.delay_fire_steps:
                a_fire = self.fire_queue[i].pop(0)

            delayed_actions.append((a_bank, a_thr, a_fire))

        # ----------------------------------------------------
        # 2. EULER INTEGRATION
        # ----------------------------------------------------
        for i in (0, 1):
            a_bank, a_thr, _ = delayed_actions[i]

            self.s[i, 4] = np.clip(a_bank * self.bank_max, -self.bank_max, self.bank_max)
            thr = np.clip(a_thr, 0.0, 1.0)

            v = self.s[i, 2]
            lift = np.cos(self.s[i, 4])  # şu an sadece drag var ama dursun
            dv = thr * 30 - self.v_drag * v * v
            self.s[i, 2] = np.clip(v + dv * self.dt, self.v_min, self.v_max)

            self._prev_bank[i] = self.s[i, 4]

            self.s[i, 3] = (self.s[i, 3] + np.tan(self.s[i, 4]) * self.dt) % (2 * np.pi)

            vx = self.s[i, 2] * np.cos(self.s[i, 3])
            vy = self.s[i, 2] * np.sin(self.s[i, 3])
            self.s[i, 0] += vx * self.dt
            self.s[i, 1] += vy * self.dt

        # ----------------------------------------------------
        # 3. REWARD + FIRE LOGIC
        # ----------------------------------------------------
        for i in (0, 1):
            j = 1 - i

            R, brg, aoff, clos = rel(i, j)

            # Predict lead error
            psi_i = self.s[i, 3]
            psi_j = self.s[j, 3]
            dx = self.s[j, 0] - self.s[i, 0]
            dy = self.s[j, 1] - self.s[i, 1]
            v_j = self.s[j, 2]

            t_hit = R / (self.bullet_speed + 1e-6)
            lead_x = dx + v_j * np.cos(psi_j) * t_hit
            lead_y = dy + v_j * np.sin(psi_j) * t_hit
            lead_bearing = (np.arctan2(lead_y, lead_x) - psi_i + np.pi) % (2 * np.pi) - np.pi
            lead_err = abs(lead_bearing)

            inside_wez = (R < self.wez_R) and (abs(brg) < self.wez_ang)
            lead_ok = (lead_err < self.lead_gate_tol)

            # ---- Shaping ----
            r[i] += 0.008 * inside_wez
            r[i] += -0.0004 * abs(aoff)
            r[i] += -0.00008 * R

            if i == 0:  # sadece RL ajanı için track shaping
                r[i] += self.track_w * np.cos(lead_err)

            # ---- FIRE GATING ----
            _, _, fire_raw = delayed_actions[i]
            fire_attempt = fire_raw > 0.5

            # cooldown countdown
            if self.fire_cd[i] > 0:
                self.fire_cd[i] -= 1

            trigger_cmd = fire_attempt

            # WEZ dışı hard gate
            if not inside_wez:
                if fire_attempt and self.ammo[i] > 0:
                    # kötü şut (WEZ dışında tetiğe basma)
                    r[i] -= self.bad_shot_w
                trigger_cmd = False

            # WEZ içinde soft quality filter
            if trigger_cmd and inside_wez:
                qual = max(0.0, np.cos(lead_err))
                # iyi şut kalitesi shaping
                r[i] += self.shot_w * qual

                # düşük kalite ise random gate
                if self.rng.random() > qual:
                    trigger_cmd = False

            # Cooldown
            if self.fire_cd[i] > 0:
                trigger_cmd = False

            # ------------------------------
            #  Real firing event
            # ------------------------------
            if trigger_cmd and self.ammo[i] > 0:
                self.ammo[i] -= 1
                self.fire_cd[i] = self.fire_cd_steps

                # ammo dependency: mermi azaldıkça ekstra ceza
                ammo_frac = self.ammo[i] / self.initial_ammo
                r[i] -= self.ammo_w * (1.0 - ammo_frac)

                if inside_wez and lead_ok:
                    # temel "iyi şut" ödülü
                    r[i] += 0.01

                    pk = self.base_pk * np.exp(-(lead_err / self.lead_gate_tol) ** 2)
                    pk = float(np.clip(pk, 0.10, 0.90))

                    if self.rng.random() < pk:
                        # isabet
                        r[i] += 0.2
                        r[j] -= 0.2
                        self.hp[j] -= 0.5

                        if self.hp[j] <= 0:
                            r[i] += 0.8
                            r[j] -= 0.8
                            done = True
                            info["winner"] = i
                else:
                    # boşa giden mermi cezası
                    r[i] -= self.bad_shot_w

        # ----------------------------------------------------
        # 4. CHECK TERMINATION
        # ----------------------------------------------------
        for i in (0, 1):
            if (
                abs(self.s[i, 0]) > self.arena
                or abs(self.s[i, 1]) > self.arena
                or self.hp[i] <= 0
            ):
                done = True

        # --- Ammo limit + zaman limiti ---
        if not done and (
            (self.ammo[0] <= 0 and self.ammo[1] <= 0)
            or (self.steps >= self.max_steps)
        ):
            done = True
            if self.timeout_as_loss:
                if "winner" not in info:
                    info["winner"] = 1  # agent0 açısından loss
                r[0] -= 0.2
                r[1] += 0.2
            else:
                # sadece hafif zaman kaybı cezası
                r[0] -= 0.05
                r[1] += 0.0

        return self._obs_all(), r, done, info
