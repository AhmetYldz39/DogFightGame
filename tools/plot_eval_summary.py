import json
import matplotlib.pyplot as plt

def plot_eval_summary(path, out_dir):
    data = json.load(open(path))

    A = sum(1 for e in data if e["winner"] == "A")
    B = sum(1 for e in data if e["winner"] == "B")
    D = sum(1 for e in data if e["winner"] == "draw")

    plt.figure()
    plt.bar(["A win", "B win", "Draw"], [A, B, D])
    plt.ylabel("Episodes")
    plt.title("Eval Outcomes")
    plt.savefig(f"{out_dir}/win_rate.png")
    plt.close()
