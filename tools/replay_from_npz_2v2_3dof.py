import os
from tools.replay_animate_2v2_3dof import animate_replay


main_dir = "C:/Users/TRON-V3/OneDrive/Masaüstü/Ders_Dosyalari/Master_Thesis/DogFightGame"
npz_path = "runs/runs_2v2/rppo_2v2_final_reward_v6/eval/replays/episode_00011_A.npz"
npz_path = os.path.join(main_dir, npz_path)

animate_replay(
    npz_path,
    save_path="episode_00011_A.mp4",
    fps=30,
)
