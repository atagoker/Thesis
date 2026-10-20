"""Retrospective evaluator. Target labels are read only in this stage."""

import csv
from datetime import datetime, timezone
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from thesislib.common import ROOT, CLASSES, BearingCNN, read_json, sha256, write_json

METHODS = ["source_only", "continued_source", "teacher_student"]
DISPLAY = {"source_only": "Source only", "continued_source": "Extra source training", "teacher_student": "EMA teacher–student", "source_ema_control": "Source-only EMA"}
COLORS = {"source_only": "#617484", "continued_source": "#d89b36", "teacher_student": "#117f89", "source_ema_control": "#8b6fa5"}


@torch.no_grad()
def probabilities(model, x, device):
    model.eval()
    return torch.cat([model(torch.from_numpy(batch).to(device)).softmax(1).cpu()
                      for batch in np.array_split(x, max(1, int(np.ceil(len(x) / 256))))]).numpy()


def metrics(y, p):
    predicted = p.argmax(1)
    confidence = p.max(1)
    correct = predicted == y
    ece = 0.0
    for lo, hi in zip(np.linspace(0, 1, 16)[:-1], np.linspace(0, 1, 16)[1:]):
        mask = (confidence >= lo) & (confidence < hi if hi < 1 else confidence <= hi)
        if mask.any():
            ece += float(mask.mean() * abs(correct[mask].mean() - confidence[mask].mean()))
    fault_truth = y != 0
    fault_predicted = predicted != 0
    fault_recall = float(fault_predicted[fault_truth].mean())
    false_positive = float(fault_predicted[~fault_truth].mean())
    return {
        "accuracy": float(accuracy_score(y, predicted)),
        "macro_f1": float(f1_score(y, predicted, labels=range(4), average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
        "fault_recall_argmax": fault_recall,
        "healthy_false_positive_rate_argmax": false_positive,
        "fault_roc_auc": float(roc_auc_score(fault_truth, 1 - p[:, 0])),
        "negative_log_likelihood": float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).mean()),
        "ece_15_bins": ece,
        "test_windows": len(y),
    }


def load_target(load, split):
    base = ROOT / "data/processed/pilot_v1"
    with np.load(base / f"load{load}_{split}_unlabeled.npz") as signals:
        x, ids = signals["x"].copy(), signals["ids"].copy()
    with np.load(base / f"evaluator/load{load}_{split}_labels.npz") as labels:
        np.testing.assert_array_equal(ids, labels["ids"])
        y = labels["y"].copy()
    return x, y, ids


def main():
    destination = ROOT / "outputs/pilot_v1"
    completed = read_json(destination / "training_complete.json")
    cfg = read_json(destination / "config_snapshot.json")
    if (destination / "source_ema_control_complete.json").exists():
        METHODS.append("source_ema_control")
        control_info = read_json(destination / "source_ema_control_complete.json")
        for item in control_info["runs"]:
            if sha256(destination / f"seed_{item['seed']}/source_ema_control.pt") != item["checkpoint_sha256"]:
                raise RuntimeError("Source EMA checkpoint changed after freeze.")
    if completed["config_sha256"] != sha256(ROOT / "configs/pilot_v1.json"):
        raise RuntimeError("Config changed after training.")
    for relative, checksum in completed["checkpoint_sha256"].items():
        if sha256(destination / relative) != checksum:
            raise RuntimeError(f"Checkpoint changed after freeze: {relative}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(4)
    results, details, confusion = [], {}, {}
    prediction_dir = destination / "predictions"
    prediction_dir.mkdir(exist_ok=True)
    source_tests = []
    for seed in cfg["seeds"]:
        folder = destination / f"seed_{seed}"
        source_checkpoint = torch.load(folder / "source_best.pt", map_location="cpu", weights_only=True)
        source_model = BearingCNN().to(device)
        source_model.load_state_dict(source_checkpoint["state_dict"])
        with np.load(ROOT / "data/processed/pilot_v1/load0_test.npz") as data:
            source_tests.append({"seed": seed, **metrics(data["y"], probabilities(source_model, data["x"], device))})
        for load in cfg["target_loads"]:
            x, y, ids = load_target(load, "test")
            for method in METHODS:
                filename = {"source_only": "source_best.pt", "continued_source": "continued_source.pt",
                            "teacher_student": f"teacher_target{load}.pt", "source_ema_control": "source_ema_control.pt"}[method]
                checkpoint = torch.load(folder / filename, map_location="cpu", weights_only=True)
                model = BearingCNN().to(device)
                model.load_state_dict(checkpoint["state_dict"])
                p = probabilities(model, x, device)
                scores = metrics(y, p)
                results.append({"seed": seed, "target_load_hp": load, "method": method, **scores})
                key = f"seed{seed}_target{load}_{method}"
                cm = confusion_matrix(y, p.argmax(1), labels=range(4))
                confusion.setdefault((load, method), []).append(cm)
                detail = {"confusion_matrix": cm.tolist(),
                          "class_report": classification_report(y, p.argmax(1), labels=range(4), target_names=CLASSES, output_dict=True, zero_division=0)}
                # Pseudo-label precision is a retrospective diagnostic only.
                fit_x, fit_y, _ = load_target(load, "fit")
                fit_p = probabilities(model, fit_x, device)
                mask = fit_p.max(1) >= cfg["confidence_threshold"]
                detail["clean_fit_pseudolabel_diagnostic"] = {
                    "coverage": float(mask.mean()), "accepted": int(mask.sum()),
                    "precision": float((fit_p.argmax(1)[mask] == fit_y[mask]).mean()) if mask.any() else None,
                    "definition": "final fixed model on unaugmented commissioning inputs; not the training trajectory",
                    "used_for_training_or_selection": False,
                }
                details[key] = detail
                np.savez_compressed(prediction_dir / f"{key}.npz", ids=ids, y_true=y, probabilities=p)
                print(f"{key}: accuracy={scores['accuracy']:.3f}, macro F1={scores['macro_f1']:.3f}", flush=True)

    frame = pd.DataFrame(results)
    frame.to_csv(destination / "results_by_seed.csv", index=False)
    pd.DataFrame(source_tests).to_csv(destination / "source_test_by_seed.csv", index=False)
    numeric = ["accuracy", "macro_f1", "balanced_accuracy", "fault_recall_argmax", "healthy_false_positive_rate_argmax", "ece_15_bins"]
    summary = frame.groupby(["target_load_hp", "method"], sort=False)[numeric].agg(["mean", "std"])
    summary.columns = ["_".join(c) for c in summary.columns]
    summary.reset_index().to_csv(destination / "results_summary.csv", index=False)
    paired = frame.pivot(index=["seed", "target_load_hp"], columns="method", values="macro_f1").reset_index()
    paired["teacher_minus_source_pp"] = 100 * (paired["teacher_student"] - paired["source_only"])
    paired["teacher_minus_continued_pp"] = 100 * (paired["teacher_student"] - paired["continued_source"])
    if "source_ema_control" in METHODS:
        paired["teacher_minus_source_ema_pp"] = 100 * (paired["teacher_student"] - paired["source_ema_control"])
    paired.to_csv(destination / "paired_changes.csv", index=False)
    write_json(destination / "evaluation.json", {
        "evaluated_utc": datetime.now(timezone.utc).isoformat(), "protocol": cfg,
        "source_tests": source_tests, "results": results, "details": details,
        "mean_paired_teacher_minus_source_pp": float(paired["teacher_minus_source_pp"].mean()),
        "mean_paired_teacher_minus_continued_pp": float(paired["teacher_minus_continued_pp"].mean()),
        "mean_paired_teacher_minus_source_ema_pp": float(paired["teacher_minus_source_ema_pp"].mean()) if "source_ema_control" in METHODS else None,
        "additional_control": "Source-only EMA was introduced after the initial three-method pilot, with unchanged settings." if "source_ema_control" in METHODS else None,
        "negative_transfer_task_seed_pairs": int((paired["teacher_minus_source_pp"] < -1e-8).sum()),
        "task_seed_pairs": len(paired),
        "code_sha256": {p.relative_to(ROOT).as_posix(): sha256(p)
                        for base in ["scripts", "thesislib", "tests"] for p in (ROOT/base).glob("*.py")},
        "interpretation": "Pilot seed variability on fixed temporal blocks, not independent-machine uncertainty.",
    })
    fig_dir = destination / "figures"
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
    positions = np.arange(3)
    width = 0.8 / len(METHODS)
    for j, method in enumerate(METHODS):
        stats = frame[frame.method == method].groupby("target_load_hp").macro_f1.agg(["mean", "std"]).loc[[1,2,3]]
        ax.bar(positions + (j-(len(METHODS)-1)/2)*width, stats["mean"]*100, width=width*0.95,
               yerr=stats["std"]*100, capsize=4, color=COLORS[method], label=DISPLAY[method])
    ax.set(xticks=positions, xticklabels=["0 → 1 hp", "0 → 2 hp", "0 → 3 hp"], ylim=(0,100),
           ylabel="Target test macro F1 (%)", title="CWRU pilot: mean ± sample SD across three training seeds")
    ax.legend(loc="lower left", fontsize=9)
    fig.savefig(fig_dir / "method_comparison.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(3, len(METHODS), figsize=(3.7*len(METHODS),10), constrained_layout=True)
    labels = ["Healthy", "Inner", "Element", "Outer"]
    for row, load in enumerate([1,2,3]):
        for column, method in enumerate(METHODS):
            counts = np.sum(confusion[(load, method)], axis=0)
            normalized = counts / counts.sum(1, keepdims=True)
            ax = axes[row,column]
            ax.imshow(normalized, vmin=0, vmax=1, cmap="Blues")
            for a in range(4):
                for b in range(4):
                    ax.text(b,a,f"{100*normalized[a,b]:.0f}%", ha="center", va="center", color="white" if normalized[a,b]>.55 else "black", fontsize=9)
            ax.set(xticks=range(4), xticklabels=labels, yticks=range(4), yticklabels=labels,
                   xlabel="Predicted class", ylabel="True class", title=f"0 → {load} hp: {DISPLAY[method]}")
    fig.suptitle("Row-normalized confusion matrices, pooled predictions across seeds")
    fig.savefig(fig_dir / "confusion_matrices.png", dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for seed in cfg["seeds"]:
        history = pd.read_csv(destination / f"seed_{seed}/source_history.csv")
        axes[0].plot(history.epoch, history.source_val_macro_f1, label=f"Seed {seed}")
    axes[0].set(xlabel="Source epoch", ylabel="Source validation macro F1", title="Source learning curves", ylim=(0,1))
    axes[0].legend(fontsize=9)
    for load in [1,2,3]:
        curves = [pd.read_csv(destination / f"seed_{s}/teacher_target{load}_history.csv").coverage.to_numpy() for s in cfg["seeds"]]
        axes[1].plot(np.arange(1, cfg["adaptation_epochs"]+1), np.mean(curves,axis=0),label=f"Target {load} hp")
    axes[1].set(xlabel="Adaptation epoch", ylabel="Accepted pseudo-label fraction", title="Mean acceptance coverage", ylim=(0,1))
    axes[1].legend(fontsize=9)
    fig.savefig(fig_dir / "training_and_coverage.png",dpi=180)
    plt.close(fig)
    print("Saved separate evaluation, prediction archives, tables, and figures.", flush=True)


if __name__ == "__main__":
    main()
