"""Evaluate frozen baseline checkpoints with separately stored reference labels."""

import csv
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, roc_auc_score)
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from thesislib.common import ROOT, BearingCNN, read_json, sha256, write_json
from thesislib.paderborn_ml import aggregate_recordings, probabilities


def metrics(y, p):
    pred = (p[:, 1] >= .5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"n": len(y), "accuracy": float(accuracy_score(y, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, labels=[0, 1], average="macro", zero_division=0)),
            "fault_recall": float(tp / (tp + fn)), "healthy_false_positive_rate": float(fp / (tn + fp)),
            "roc_auc": float(roc_auc_score(y, p[:, 1])),
            "fault_average_precision": float(average_precision_score(y, p[:, 1])),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    out = ROOT / "outputs/paderborn_baseline_v1"
    completed = read_json(out / "training_complete.json")
    cfg = read_json(ROOT / "configs/paderborn_baseline_v1.json")
    prepared = read_json(ROOT / "data/manifests/paderborn_v1_summary.json")
    if sha256(ROOT / "configs/paderborn_baseline_v1.json") != completed["config_sha256"]:
        raise ValueError("Configuration differs from frozen training.")
    if sha256(ROOT / "data/manifests/paderborn_v1_summary.json") != completed["prepared_summary_sha256"]:
        raise ValueError("Preparation evidence changed after training.")
    for path, digest in completed["implementation_sha256"].items():
        if sha256(ROOT / path) != digest:
            raise ValueError(f"Training implementation changed: {path}")
    for path, digest in completed["checkpoint_sha256"].items():
        if sha256(out / path) != digest:
            raise ValueError(f"Checkpoint changed: {path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = ROOT / "data/processed/paderborn_v1"
    conditions = [cfg["source_condition"], *cfg["target_conditions"]]
    rows, bearing_rows, pooled = [], [], {c: [] for c in conditions}
    prediction_dir = out / "predictions"
    prediction_dir.mkdir(exist_ok=True)
    for condition in conditions:
        key = f"evaluation_{condition}"
        data_path = dataset / f"{key}_unlabeled.npz"
        label_path = dataset / "evaluator" / f"{key}_labels.npz"
        for path in [data_path, label_path]:
            if sha256(path) != prepared["prepared_sha256"][str(path.relative_to(dataset))]:
                raise ValueError(f"Evaluation input changed: {path}")
        with np.load(data_path, allow_pickle=False) as data, np.load(label_path, allow_pickle=False) as labels:
            if set(data.files) != {"x", "ids"} or not np.array_equal(data["ids"], labels["ids"]):
                raise ValueError("Evaluation schema/label alignment failure.")
            x, ids, y = torch.from_numpy(data["x"].copy()), data["ids"].copy(), labels["y"].copy()
        for seed in cfg["seeds"]:
            checkpoint = torch.load(out / f"seed_{seed}/source_best.pt", map_location="cpu", weights_only=True)
            model = BearingCNN(num_classes=2).to(device)
            model.load_state_dict(checkpoint["state_dict"])
            p = probabilities(model, x, device)
            records, rp, ry = aggregate_recordings(p, ids, y)
            for level, truth, prob in [("window", y, p), ("recording", ry, rp)]:
                rows.append({"condition": condition, "seed": seed, "level": level, **metrics(truth, prob)})
            np.savez_compressed(prediction_dir / f"{condition}_seed{seed}.npz", ids=records, y=ry, probabilities=rp)
            bearing_ids = np.asarray([record.split("_")[3] for record in records])
            if set(bearing_ids) != set(cfg["bearing_groups"]["evaluation"]):
                raise ValueError("Unexpected evaluation bearing identities.")
            for bearing in sorted(set(bearing_ids)):
                mask = bearing_ids == bearing
                bearing_rows.append({"condition": condition, "seed": seed, "bearing": bearing,
                                     "class_id": int(ry[mask][0]), "recordings": int(mask.sum()),
                                     "mean_fault_probability": float(rp[mask, 1].mean()),
                                     "recording_accuracy": float(((rp[mask, 1] >= .5) == ry[mask]).mean())})
            pooled[condition].append(confusion_matrix(ry, rp[:, 1] >= .5, labels=[0, 1]))
            print(f"{condition} seed {seed}: recording macro F1={rows[-1]['macro_f1']:.4f}; "
                  f"balanced accuracy={rows[-1]['balanced_accuracy']:.4f}", flush=True)
    write_csv(out / "results_by_seed.csv", rows)
    write_csv(out / "per_bearing_results.csv", bearing_rows)
    metric_names = ["accuracy", "balanced_accuracy", "macro_f1", "fault_recall",
                    "healthy_false_positive_rate", "roc_auc", "fault_average_precision"]
    summary = []
    for condition in conditions:
        group = [row for row in rows if row["condition"] == condition and row["level"] == "recording"]
        result = {"condition": condition, "recordings_per_seed": group[0]["n"]}
        for name in metric_names:
            values = [row[name] for row in group]
            result[name + "_mean"] = float(np.mean(values))
            result[name + "_sd"] = float(np.std(values, ddof=1))
        summary.append(result)
    write_csv(out / "results_summary.csv", summary)
    figures = out / "figures"
    figures.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 4, figsize=(15, 4), constrained_layout=True)
    for ax, condition in zip(axes, conditions):
        cm = np.sum(pooled[condition], axis=0)
        ax.imshow(cm / cm.sum(axis=1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]}\n({cm[i,j]/cm[i].sum():.1%})", ha="center", va="center",
                        color="white" if cm[i, j] / cm[i].sum() > .55 else "black")
        ax.set(title=condition, xlabel="Predicted", ylabel="True",
               xticks=[0, 1], yticks=[0, 1], xticklabels=cfg["classes"], yticklabels=cfg["classes"])
    fig.suptitle("Source-only Paderborn baseline: pooled seed predictions (not independent samples)")
    fig.savefig(figures / "confusion_matrices.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    ax.bar(range(4), [r["macro_f1_mean"] * 100 for r in summary], color="#157f83", alpha=.6, label="mean")
    for offset, seed in zip([-.12, 0, .12], cfg["seeds"]):
        values = [next(r["macro_f1"] * 100 for r in rows if r["condition"] == condition
                       and r["seed"] == seed and r["level"] == "recording") for condition in conditions]
        ax.scatter(np.arange(4) + offset, values, s=35, label=f"seed {seed}", zorder=3)
    ax.set(xticks=range(4), xticklabels=conditions, ylim=(0, 105), ylabel="Recording macro F1 (%)",
           title="All three training seeds and their mean; held-out bearings fixed")
    ax.legend()
    fig.savefig(figures / "baseline_macro_f1.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    first_seed = cfg["seeds"][0]
    for ax, condition in zip(axes.flat, conditions):
        with np.load(prediction_dir / f"{condition}_seed{first_seed}.npz") as data:
            y, p = data["y"], data["probabilities"][:, 1]
        for label, color in [(0, "#157f83"), (1, "#d97d29")]:
            ax.hist(p[y == label], bins=np.linspace(0, 1, 21), alpha=.8, histtype="step", linewidth=2,
                    label=cfg["classes"][label], color=color)
        ax.axvline(.5, color="black", ls="--", label="fixed decision threshold")
        ax.set(title=condition, xlabel="Mean faulty model score (not calibrated risk)", ylabel="Recordings", xlim=(0, 1))
        ax.legend(fontsize=8)
    fig.suptitle(f"Why score ranking and threshold decisions differ: fixed first seed {first_seed}")
    fig.savefig(figures / "recording_scores_seed7.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for seed in cfg["seeds"]:
        with (out / f"seed_{seed}/source_history.csv").open(newline="", encoding="utf-8") as stream:
            history = list(csv.DictReader(stream))
        epoch = [int(row["epoch"]) for row in history]
        axes[0].plot(epoch, [float(row["source_loss"]) for row in history], label=f"seed {seed}")
        axes[1].plot(epoch, [float(row["source_val_recording_macro_f1"]) for row in history], label=f"seed {seed}")
    axes[0].set(xlabel="Epoch", ylabel="Source training loss", title="Fit to training bearings")
    axes[1].set(xlabel="Epoch", ylabel="Recording macro F1", title="Validation on different source-condition bearings", ylim=(0, 1))
    for ax in axes:
        ax.legend()
    fig.savefig(figures / "source_training_history.png", dpi=160)
    plt.close(fig)
    write_json(out / "evaluation.json", {"checkpoint_hashes_verified": True,
               "target_labels_used_for_training_or_selection": False,
               "level_for_primary_reporting": "recording", "summary": summary,
               "evaluation_bearings": cfg["bearing_groups"]["evaluation"],
               "number_of_independent_healthy_evaluation_bearings": 1,
               "limitations": prepared["limitations"]})
    lines = ["# Paderborn source-only baseline: measured results", "",
             "This is a binary healthy/faulty experiment, not a teacher-student result.", "",
             "## Recording-level results", "",
             "Mean +/- sample standard deviation across three training seeds. Same fixed held-out bearings in all runs.", "",
             "| Condition | Accuracy (%) | Balanced accuracy (%) | Macro F1 (%) | Fault recall (%) | Healthy FPR (%) |",
             "|---|---:|---:|---:|---:|---:|"]
    for row in summary:
        cells = [f"{100*row[k+'_mean']:.2f} +/- {100*row[k+'_sd']:.2f}" for k in metric_names[:5]]
        lines.append("| " + row["condition"] + " | " + " | ".join(cells) + " |")
    lines += ["", "## What was actually done", "",
              f"- Attempted numerical audit of {prepared['raw_recordings_audit_attempted']} original recordings; "
              f"{prepared['readable_valid_recordings']} readable/valid; {len(prepared['excluded_recordings'])} excluded with logged reasons.",
              f"- Selected {prepared['selected_recordings']} recordings under the predefined bearing/condition split.",
              "- Trained only on source fit; selected checkpoints only on source validation.",
              "- Source-only training blocks file access to prepared target/evaluator archives with a Python audit hook.",
              "- Froze and hashed all three checkpoints before opening evaluator reference labels.",
              "- Reserved target-adaptation inputs contain x and IDs only and were not used by this baseline.", "",
              "## How to interpret this", "",
              "Each condition has 120 evaluation recordings: 20 healthy and 100 faulty, from six bearings. "
              "An always-faulty classifier would achieve 83.33% accuracy, 50% balanced accuracy, and 45.45% macro F1. "
              "Accuracy alone is therefore inadequate. There is only ONE independent healthy test bearing. "
              "Repetitions and windows are correlated; the seed standard deviation is not a population confidence interval.", "",
              "The source-condition row tests unseen bearings without changing the nominal operating condition. "
              "Target rows additionally change operating conditions. This is not a clean estimate of condition shift "
              "alone on source training bearings, nor an industrial deployment validation.", "",
              "ROC AUC is much higher than fixed-threshold performance for several tasks (see results_summary.csv). "
              "A high ranking metric does not mean a .5 threshold works: many fault scores fall below that threshold. "
              "This suggests a score/decision shift worth investigating, not proof that calibration alone solves "
              "cross-bearing generalization. Never choose a new threshold using these target reference labels.", "",
              "These frozen test results are now observed. Do not tune v1 settings against them; declare a new "
              "experiment for any changes. Teacher-student adaptation still needs implementation and matched controls.", "",
              "## Preparation limitations", ""]
    lines += [f"The embedded-axis mean-rate diagnostic (assuming seconds) differs from nominal by "
              f"over 1% in {prepared['records_over_1_percent_embedded_rate_difference']} recordings; "
              f"maximum absolute discrepancy is {100*prepared['maximum_abs_embedded_rate_difference_fraction']:.2f}%. "
              "This is a metadata inconsistency, not a verified alternative acquisition rate.", ""]
    lines += ["- " + limitation for limitation in prepared["limitations"]]
    lines += ["", "## Excluded files", ""]
    lines += [f"- `{row['file']}`: {row['error']}; selected by protocol: {row['would_have_been_selected']}"
              for row in prepared["excluded_recordings"]]
    lines += ["", "## Evidence", "",
              "- Protocol: `reports/PADERBORN_PROTOCOL_V1.md`.",
              "- Exact configuration: `configs/paderborn_baseline_v1.json`.",
              "- Per-record raw checksums and quality: `data/manifests/paderborn_v1_records.csv`.",
              "- Per-seed window and recording metrics: `outputs/paderborn_baseline_v1/results_by_seed.csv`.",
              "- Per-bearing breakdown: `outputs/paderborn_baseline_v1/per_bearing_results.csv`.",
              "- Frozen model evidence: `outputs/paderborn_baseline_v1/training_complete.json`.",
              "- Official data reference: [Lessmeier et al. (2016)](https://mb.uni-paderborn.de/fileadmin-mb/kat/PDF/Veroeffentlichungen/20160703_PHME16_CM_bearing.pdf).", ""]
    report = ROOT / "reports/PADERBORN_BASELINE_RESULTS.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"EVALUATION COMPLETE. Report: {report}")


if __name__ == "__main__":
    main()
