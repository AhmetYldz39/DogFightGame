# norm_utils.py
import pickle
import numpy as np

def make_norm_obs_fn_from_vec(vec_path: str):
    with open(vec_path, "rb") as f:
        obj = pickle.load(f)

    obs_rms = getattr(obj, "obs_rms", None)
    clip_obs = getattr(obj, "clip_obs", 10.0)

    if obs_rms is None:
        raise ValueError("obs_rms not found in vecnormalize.pkl")

    mean = np.array(obs_rms.mean, dtype=np.float32)
    var  = np.array(obs_rms.var,  dtype=np.float32)
    eps  = 1e-8

    def norm_obs(o: np.ndarray) -> np.ndarray:
        o = o.astype(np.float32)
        o = (o - mean) / np.sqrt(var + eps)
        return np.clip(o, -clip_obs, clip_obs)

    return norm_obs
