import numpy as np
import os
import sys
from sb3_contrib import RecurrentPPO

from scenarios.scenario_1v1.dogfight_wrappers import DogfightSoloEnvPool
from tools.replay_recorder import render_eval_episode
from tools.norm_utils import make_norm_obs_fn_from_vec

# =========================
# PATHS
# =========================
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

AGENT_MODEL = os.path.join(PROJECT_ROOT, "runs/runs_1v1/rppo_sepLSTM_heur_20251206_234439/fine_tune_model_v3.zip")
AGENT_VEC = os.path.join(PROJECT_ROOT, "runs/runs_1v1/rppo_sepLSTM_heur_20251206_234439/vecnormalize.pkl")

FROZEN_MODEL = os.path.join(PROJECT_ROOT, "runs/runs_1v1/rppo_sepLSTM_heur_20251122_210239/frozen_expert_v1.zip")
FROZEN_VEC = os.path.join(PROJECT_ROOT, "runs/runs_1v1/rppo_sepLSTM_heur_20251122_210239/vecnormalize.pkl")

OUT_GIF = "best_episode_1v1_frozen.gif"
N_EVAL_EPISODES = 10

# =========================
# ENV FACTORY
# =========================
def make_env_frozen():
    return DogfightSoloEnvPool(
        seed=0,
        timeout_as_loss=True,
        opponent_pool=[
            ("heuristic")
        ],
        pool_probs=[1.0],
    )

# =========================
# LOAD MODEL + NORM
# =========================
dummy_env = make_env_frozen()
model = RecurrentPPO.load(AGENT_MODEL, env=dummy_env)
norm_obs_fn = make_norm_obs_fn_from_vec(AGENT_VEC)

best_score = -1e9

print("🔍 Evaluating against frozen expert...\n")

# =========================
# EVALUATION LOOP
# =========================
for ep in range(N_EVAL_EPISODES):
    info = render_eval_episode(
        model=model,
        make_env_fn=make_env_frozen,
        save_path=f"_tmp_{ep}.gif",
        norm_obs_fn=norm_obs_fn,
        deterministic=True,
    )

    winner = info.get("winner", None)

    if winner != 0:
        print(f"Episode {ep:02d} | LOSS")
        continue

    time_to_kill = info.get("time", 999.0)
    score = 1000.0 - time_to_kill

    print(f"Episode {ep:02d} | WIN | t={time_to_kill:.1f}")

    if score > best_score:
        best_score = score
        best_ep = ep

# =========================
# FINAL RENDER
# =========================
print("\n🎥 Rendering BEST episode...\n")

render_eval_episode(
    model=model,
    make_env_fn=make_env_frozen,
    save_path=OUT_GIF,
    norm_obs_fn=norm_obs_fn,
    deterministic=True,
)

print("===================================")
print(f"✅ Best episode index : {best_ep}")
print(f"📁 Saved as           : {OUT_GIF}")
print("===================================")
