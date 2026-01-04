import numpy as np


# =========================================================
# Heuristic opponent (3DOF)
# =========================================================
class HeuristicBot3DOF:
    def __init__(self):
        self.nz_cruise = 1.0
        self.nz_turn = 2.5
        self.mu_max = np.deg2rad(45)

        self.fire_range = 900.0
        self.fire_brg = np.deg2rad(20)
        self.fire_aoff = np.deg2rad(20)

    def act(self, obs: np.ndarray) -> np.ndarray:
        enemy0 = obs[5 + 6: 5 + 12]

        R = 4000.0 * np.clip(enemy0[0], 0.0, 1.0)
        brg = np.arctan2(enemy0[1], enemy0[2])
        aoff = np.arctan2(enemy0[3], enemy0[4])

        if abs(brg) > np.deg2rad(5):
            mu = np.clip(np.sign(brg) * self.mu_max, -self.mu_max, self.mu_max)
            nz = self.nz_turn
        else:
            mu = 0.0
            nz = self.nz_cruise

        nx = 0.0

        fire = 0.0
        if (
            R < self.fire_range
            and abs(brg) < self.fire_brg
            and abs(aoff) < self.fire_aoff
        ):
            fire = 1.0

        return np.array([nx, nz, mu, fire], dtype=np.float32)
