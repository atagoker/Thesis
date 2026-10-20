"""Retrospective per-class pseudo-label diagnostic after model freeze."""

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from thesislib.common import ROOT, CLASSES, BearingCNN, read_json
from evaluate_pilot import load_target, probabilities


def main():
    destination = ROOT / "outputs/pilot_v1"
    cfg = read_json(destination / "config_snapshot.json")
    device = torch.device("cuda")
    torch.set_num_threads(4)
    rows = []
    for seed in cfg["seeds"]:
        for load in cfg["target_loads"]:
            x, y, _ = load_target(load, "fit")
            for role, filename in [("source_only", "source_best.pt"), ("teacher_student", f"teacher_target{load}.pt")]:
                saved = torch.load(destination / f"seed_{seed}" / filename, map_location="cpu", weights_only=True)
                model = BearingCNN().to(device)
                model.load_state_dict(saved["state_dict"])
                p = probabilities(model, x, device)
                selected, prediction = p.max(1) >= cfg["confidence_threshold"], p.argmax(1)
                for c, name in enumerate(CLASSES):
                    true_class = y == c
                    accepted = selected & true_class
                    rows.append({"seed": seed, "target_load_hp": load, "model": role,
                                 "true_class": name, "support": int(true_class.sum()),
                                 "accepted": int(accepted.sum()),
                                 "coverage": float(accepted.sum()/true_class.sum()),
                                 "accepted_precision": float((prediction[accepted]==y[accepted]).mean()) if accepted.any() else None})
    frame = pd.DataFrame(rows)
    frame.to_csv(destination / "pseudolabel_diagnostics_by_class.csv", index=False)
    table = frame.groupby(["model", "true_class"])[["coverage", "accepted_precision"]].mean()
    table.to_csv(destination / "pseudolabel_diagnostics_summary.csv")
    print(table.to_string(), flush=True)
    fig, axes = plt.subplots(1,3,figsize=(12,4.3),constrained_layout=True)
    for ax, load in zip(axes,[1,2,3]):
        for role, color, offset, label in [("source_only", "#617484", -0.18, "Source only"),
                                            ("teacher_student", "#117f89", 0.18, "Final teacher")]:
            group = frame[(frame.model==role)&(frame.target_load_hp==load)].groupby("true_class").coverage.mean().reindex(CLASSES)
            ax.bar(np.arange(4)+offset, group.to_numpy()*100,width=0.35,color=color,label=label)
        ax.set(xticks=range(4),xticklabels=["Healthy","Inner","Element","Outer"],ylim=(0,100),
               xlabel="True class (evaluator only)",ylabel="Accepted windows (%)",title=f"Target {load} hp")
        ax.tick_params(axis="x",labelrotation=25)
    axes[0].legend(fontsize=9)
    fig.suptitle("Retrospective clean-input confidence coverage at threshold 0.90")
    fig.savefig(destination / "figures/pseudolabel_coverage_by_class.png",dpi=170)
    plt.close(fig)
    print("This analysis used held-back commissioning labels after training and did not change any model.",flush=True)


if __name__ == "__main__":
    main()
