import torch
import pandas as pd
import numpy as np


main_dir = 'runs/rppo_lstm_pool_20251109_145454/final_model/'

# model file
model_file = main_dir + 'policy.pth'

SD = torch.load(model_file, map_location="cpu")
rows = []
for n, w in SD.items():
    if w.dtype.is_floating_point:
        a = w.detach().cpu().float().view(-1)
        rows.append(dict(param=n, shape=str(tuple(w.shape)),
                         mean=float(a.mean()), std=float(a.std()),
                         min=float(a.min()), max=float(a.max()),
                         numel=a.numel()))
df = pd.DataFrame(rows).sort_values("param")
print(df.to_string(index=False))
# df.to_csv("policy_summary.csv", index=False)


# optimization file
opt_file = main_dir + 'policy.optimizer.pth'

OPT = torch.load(opt_file, map_location="cpu")
print("lr =", OPT["param_groups"][0]["lr"])
steps = [s.get("step", 0) for s in OPT["state"].values() if isinstance(s, dict)]
print("med_step =", np.median(steps) if steps else None)
