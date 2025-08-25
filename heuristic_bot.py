# heuristic_bot.py
import numpy as np

def heuristic_policy(obs):
    brg = np.pi * obs[1]
    aoff = np.pi * obs[2]
    R = 2000.0 * obs[0]

    if brg > np.deg2rad(3): bank_cmd = 2
    elif brg < -np.deg2rad(3): bank_cmd = 0
    else: bank_cmd = 1

    clos = 200.0 * np.arctanh(obs[3])
    if clos < 0: thr_cmd = 2
    elif clos > 120: thr_cmd = 0
    else: thr_cmd = 1

    inside_wez = (R < 600.0) and (abs(brg) < np.deg2rad(20))
    fire = 1 if (inside_wez and abs(aoff) < np.deg2rad(15)) else 0

    return (bank_cmd, thr_cmd, fire)
