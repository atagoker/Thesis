"""Retrospective pseudo-label audit, figures and measured report after model freeze."""

import csv
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from thesislib.common import ROOT, read_json, sha256, write_json
from thesislib.commissioning import write_csv
from thesislib.commissioning_evaluation import evaluation_arrays, verify_frozen_run

LABELS = {"source_only": "Initial source", "continued_source": "Extra source", "source_ema": "Source EMA",
          "teacher_student": "Teacher-student"}
TARGET_LABELS = {"N09_M07_F10": "Reduced speed", "N15_M01_F10": "Reduced torque", "N15_M07_F04": "Reduced radial force"}
COLORS = ["#60758a", "#d59639", "#438b93", "#8b5bb0"]
SEED_COLORS = ["#1877a3", "#c47720", "#8454a7"]


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def describe_trace(trace, y, epoch):
    chosen = trace["epoch"] == epoch
    truth = y[trace["index"][chosen]]
    pseudo = trace["pseudo"][chosen]
    accepted = trace["accepted"][chosen]
    confidence = trace["confidence"][chosen]
    count = int(accepted.sum())
    correct = pseudo == truth
    row = {"epoch": epoch, "exposures": int(chosen.sum()), "accepted": count,
           "acceptance": float(accepted.mean()), "accepted_precision": ratio(int((correct & accepted).sum()), count),
           "all_pseudo_accuracy": float(correct.mean()), "mean_confidence": float(confidence.mean()),
           "accepted_wrong": int((~correct & accepted).sum()),
           "accepted_faulty_share": ratio(int(((pseudo == 1) & accepted).sum()), count),
           "accepted_mean_confidence": float(confidence[accepted].mean()) if count else None}
    for k, label in enumerate(["healthy", "faulty"]):
        true_mask, predicted_mask = truth == k, pseudo == k
        row[f"true_{label}_exposures"] = int(true_mask.sum())
        row[f"true_{label}_coverage"] = ratio(int((true_mask & accepted).sum()), int(true_mask.sum()))
        row[f"accepted_predicted_{label}"] = int((predicted_mask & accepted).sum())
        row[f"accepted_{label}_precision"] = ratio(int((correct & predicted_mask & accepted).sum()),
                                                   int((predicted_mask & accepted).sum()))
    return row


def save_figure(fig, directory, name):
    fig.savefig(directory / name, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def figures(cfg, out, evaluation, offline):
    directory = out / "figures"
    directory.mkdir(exist_ok=True)
    rows = read_csv(out / "results_by_seed.csv")
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), sharey=True, layout="constrained")
    for ax, target in zip(axes, cfg["target_conditions"]):
        for i, method in enumerate(cfg["methods"]):
            vals = [float(r["macro_f1"]) * 100 for r in rows if r["scope"] == "target" and r["level"] == "recording"
                    and r["adaptation_target"] == target and r["method"] == method]
            ax.bar(i, np.mean(vals), color=COLORS[i], alpha=.65, width=.72)
            ax.scatter(i + np.linspace(-.14, .14, 3), vals, c=SEED_COLORS, s=28, zorder=3, edgecolor="white", linewidth=.4)
            ax.text(i, np.mean(vals) + 2, f"{np.mean(vals):.1f}", ha="center", fontsize=9)
        ax.set_xticks(range(4), [LABELS[m] for m in cfg["methods"]], rotation=25, ha="right")
        ax.set_title(f"{TARGET_LABELS[target]}\n{target}", fontsize=11)
        ax.set_ylim(0, 110)
        ax.set_yticks(range(0, 101, 20))
        ax.grid(axis="y", alpha=.2)
    axes[0].set_ylabel("Recording macro F1 (%)")
    for seed, color in zip(cfg["seeds"], SEED_COLORS):
        axes[0].scatter([], [], c=color, s=28, label=f"seed {seed}")
    axes[0].legend(loc="lower right", fontsize=8)
    fig.suptitle("Paderborn: fixed-split exploratory comparison | bars = mean; dots = all 3 seeds", fontsize=13)
    save_figure(fig, directory, "comparison_macro_f1.png")

    pairs = [r for r in read_csv(out / "paired_comparisons.csv") if r["control"] == "source_ema"]
    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
    for i, row in enumerate(pairs):
        value = float(row["delta_macro_f1_pp"])
        ax.plot([0, value], [i, i], color="#b4bdc7", linewidth=3)
        ax.scatter(value, i, color=SEED_COLORS[cfg["seeds"].index(int(row["seed"]))], s=45, zorder=3)
        ax.annotate(f"{value:+.2f}", (value, i), xytext=(7 if value >= 0 else -7, 0), textcoords="offset points",
                    ha="left" if value >= 0 else "right", va="center", fontsize=9)
    ax.set_yticks(range(len(pairs)), [f"{TARGET_LABELS[r['target']]} | seed {r['seed']}" for r in pairs])
    ax.axvline(0, color="#394957", linestyle="--", linewidth=1)
    ax.margins(x=.25)
    ax.set_xlabel("Teacher-student minus source EMA: recording macro F1 (percentage points)")
    ax.set_title("Value added by unlabeled-target training: all nine paired comparisons")
    ax.invert_yaxis()
    save_figure(fig, directory, "paired_deltas.png")

    fig, axes = plt.subplots(3, 3, figsize=(13, 9), sharex=True, layout="constrained")
    columns = [("acceptance", "Accepted target exposures"), ("accepted_precision", "Accepted pseudo-label precision"),
               ("accepted_faulty_share", "Faulty share of accepted labels")]
    for row_index, target in enumerate(cfg["target_conditions"]):
        for seed_index, seed in enumerate(cfg["seeds"]):
            subset = [r for r in offline if r["target"] == target and r["seed"] == seed]
            for col, (key, title) in enumerate(columns):
                ax = axes[row_index, col]
                ax.plot([r["epoch"] for r in subset], [np.nan if r[key] is None else r[key] * 100 for r in subset],
                        color=SEED_COLORS[seed_index], label=f"seed {seed}")
                ax.set_ylim(-3, 103)
                ax.grid(alpha=.2)
                if row_index == 0:
                    ax.set_title(title, fontsize=11)
                if row_index == 2:
                    ax.set_xlabel("Epoch")
            axes[row_index, 0].set_ylabel(TARGET_LABELS[target] + "\nPercent")
    axes[0, 0].legend(fontsize=9)
    fig.suptitle("Actual training exposures | precision uses reference labels only AFTER training", fontsize=13)
    save_figure(fig, directory, "pseudo_label_diagnostics.png")

    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5), sharex=True, layout="constrained")
    for col, target in enumerate(cfg["target_conditions"]):
        for seed_index, seed in enumerate(cfg["seeds"]):
            history = read_csv(out / f"seed_{seed}" / target / "history.csv")
            epoch = [int(r["epoch"]) for r in history]
            axes[0, col].plot(epoch, [float(r["teacher_student_disagreement"]) * 100 for r in history],
                              color=SEED_COLORS[seed_index], label=f"seed {seed}")
            axes[1, col].plot(epoch, [float(r["source_val_teacher_macro_f1"]) * 100 for r in history], color=SEED_COLORS[seed_index])
        axes[0, col].set_title(TARGET_LABELS[target])
        axes[0, col].set_ylim(0, 15)
        axes[1, col].set_ylim(0, 100)
        axes[1, col].set_xlabel("Epoch")
        for ax in axes[:, col]:
            ax.grid(alpha=.2)
    axes[0, 0].set_ylabel("Weak teacher / strong student\ndisagreement (%)")
    axes[1, 0].set_ylabel("Source validation teacher\nrecording macro F1 (%)")
    axes[0, 0].legend(fontsize=9)
    fig.suptitle("Learning diagnostics: agreement is not evidence of correctness", fontsize=13)
    save_figure(fig, directory, "consistency_and_validation.png")

    fig, axes = plt.subplots(3, 4, figsize=(13, 9.8), layout="constrained")
    for i, target in enumerate(cfg["target_conditions"]):
        for j, method in enumerate(cfg["methods"]):
            selected = [r for r in rows if r["scope"] == "target" and r["level"] == "recording"
                        and r["adaptation_target"] == target and r["method"] == method]
            cm = np.array([[sum(int(r[k]) for r in selected) for k in ["tn", "fp"]],
                           [sum(int(r[k]) for r in selected) for k in ["fn", "tp"]]])
            ax = axes[i, j]
            ax.imshow(cm, cmap="Blues", vmin=0, vmax=300)
            for (rr, cc), value in np.ndenumerate(cm):
                ax.text(cc, rr, str(value), ha="center", va="center", color="white" if value > 150 else "#17222d")
            ax.set_xticks([0, 1], ["Healthy", "Faulty"], fontsize=9)
            ax.set_yticks([0, 1], ["Healthy", "Faulty"], fontsize=9)
            if i == 0:
                ax.set_title(LABELS[method])
            if j == 0:
                ax.set_ylabel(TARGET_LABELS[target] + "\nTrue class", fontsize=10)
            if i == 2:
                ax.set_xlabel("Predicted class")
    fig.suptitle("Confusion counts summed over 3 seeds (same 120 recordings repeated; not 360 independent recordings)", fontsize=11)
    save_figure(fig, directory, "confusion_matrices.png")

    bearing_rows = read_csv(out / "per_bearing_results.csv")
    fig, axes = plt.subplots(1, 3, figsize=(13, 5), layout="constrained")
    for ax, target in zip(axes, cfg["target_conditions"]):
        bearings = sorted({r["bearing"] for r in bearing_rows if r["class_id"] == "1"})
        matrix = np.array([[np.mean([float(r["accuracy"]) for r in bearing_rows if r["scope"] == "target"
                                    and r["adaptation_target"] == target and r["bearing"] == bearing and r["method"] == method])
                            for method in cfg["methods"]] for bearing in bearings])
        ax.imshow(matrix, vmin=0, vmax=1, cmap="Blues", aspect="auto")
        for (rr, cc), value in np.ndenumerate(matrix):
            ax.text(cc, rr, f"{value*100:.0f}", ha="center", va="center", color="white" if value > .55 else "#17222d")
        ax.set_yticks(range(len(bearings)), bearings)
        ax.set_xticks(range(4), [LABELS[m] for m in cfg["methods"]], rotation=30, ha="right")
        ax.set_title(TARGET_LABELS[target])
    fig.suptitle("Fault recall (%) by held-out faulty bearing | mean across seeds; 20 recordings per bearing/seed", fontsize=12)
    save_figure(fig, directory, "per_bearing_fault_recall.png")


def main():
    cfg, baseline, prepared, out, frozen = verify_frozen_run()
    evaluation = read_json(out / "evaluation.json")
    if evaluation["training_completion_sha256"] != sha256(out / "training_complete.json"):
        raise ValueError("Evaluation does not belong to this frozen run.")
    offline, final, training = [], [], []
    for target in cfg["target_conditions"]:
        _, ids, y = evaluation_arrays(cfg, prepared, f"target_adaptation_{target}")
        for seed in cfg["seeds"]:
            folder = out / f"seed_{seed}" / target
            history = read_csv(folder / "history.csv")
            training.extend({"target": target, "seed": seed, **row} for row in history)
            with np.load(folder / "pseudo_trace.npz", allow_pickle=False) as archive:
                trace = {key: archive[key].copy() for key in archive.files}
            if np.any(trace["index"] < 0) or np.any(trace["index"] >= len(y)):
                raise ValueError("Trace window index out of range.")
            if not np.array_equal(trace["accepted"], trace["confidence"] >= cfg["confidence_threshold"]):
                raise ValueError("Trace confidence mask inconsistent with declared threshold.")
            for epoch in range(1, cfg["adaptation_epochs"] + 1):
                row = {"target": target, "seed": seed, **describe_trace(trace, y, epoch)}
                logged = history[epoch - 1]
                if row["exposures"] != int(logged["target_exposures"]) or row["accepted"] != int(logged["accepted"]):
                    raise ValueError("Trace and online history disagree.")
                offline.append(row)
                if epoch == cfg["adaptation_epochs"]:
                    final.append(row)
    write_csv(out / "pseudo_labels_by_epoch.csv", offline)
    write_csv(out / "pseudo_labels_final_epoch.csv", final)
    write_csv(out / "training_diagnostics.csv", training)
    figures(cfg, out, evaluation, offline)
    analysis = {"scope": "Retrospective actual training exposures; no feedback to model selection",
                "analysis_script_sha256": sha256(Path(__file__)),
                "all_trace_histories_reconciled": True, "reference_labels_used_only_after_freeze": True,
                "final_epoch": final,
                "accepted_precision_range": [min(r["accepted_precision"] for r in final if r["accepted_precision"] is not None),
                                             max(r["accepted_precision"] for r in final if r["accepted_precision"] is not None)],
                "notes": ["Repeated target windows within epochs are repeated exposures, not independent measurements.",
                          "Class concentration is not by itself proof of error; refer to retrospective precision.",
                          "This version does not fit or validate a calibration correction."]}
    write_json(out / "analysis.json", analysis)
    lines = ["# Paderborn commissioning: measured results", "", "Generated from frozen experiment evidence. Exploratory fixed-split study.",
             "", "## Main result", "", "Mean recording macro F1 across three targets and three seeds:", "",
             "| Method | Mean macro F1 (%) |", "|---|---:|"]
    for method, value in evaluation["overall_target_macro_f1"].items():
        lines.append(f"| {LABELS[method]} | {value*100:.2f} |")
    lines.extend(["", "Paired differences use teacher-student minus each control; unit is percentage points.", "",
                  "| Control | Mean difference | Improved / worse / tied (of 9) |", "|---|---:|---:|"])
    for r in evaluation["comparisons"]:
        lines.append(f"| {LABELS[r['control']]} | {r['mean_delta_macro_f1_pp']:+.2f} | {r['improved']} / {r['worse']} / {r['tied']} |")
    lines.extend(["", "Source EMA is the primary comparator. Averaging these nine runs is descriptive; the same bearings",
                  "and seed-specific source models recur across conditions. It is not a population confidence interval.",
                  "", "## All target conditions", "", "Values are mean ± sample SD across three seeds, in percent.", "",
                  "| Condition | Method | Macro F1 | Balanced accuracy | Fault recall | Healthy FPR | Accuracy |",
                  "|---|---|---:|---:|---:|---:|---:|"])
    for r in evaluation["summary"]:
        if r["scope"] != "target":
            continue
        cells = [f"{r[k+'_mean']*100:.2f} ± {r[k+'_sd']*100:.2f}" for k in
                 ["macro_f1", "balanced_accuracy", "fault_recall", "healthy_false_positive_rate", "accuracy"]]
        lines.append(f"| {r['adaptation_target']} | {LABELS[r['method']]} | " + " | ".join(cells) + " |")
    lines.extend(["", "## Actual pseudo-label decisions in the final epoch", "",
                  "Coverage and precision below count the repeated exposures used during training, not unique bearings.",
                  "Reference labels were joined only after every checkpoint was frozen. None of these numbers selected a model.", "",
                  "| Target | Seed | Accepted (%) | Accepted precision (%) | Accepted faulty share (%) | True healthy coverage (%) | True faulty coverage (%) |",
                  "|---|---:|---:|---:|---:|---:|---:|"])
    for r in final:
        values = ["n/a" if r[k] is None else f"{r[k]*100:.2f}" for k in
                  ["acceptance", "accepted_precision", "accepted_faulty_share", "true_healthy_coverage", "true_faulty_coverage"]]
        lines.append(f"| {r['target']} | {r['seed']} | " + " | ".join(values) + " |")
    lines.extend(["", "## Figures", ""])
    for name, caption in [("comparison_macro_f1.png", "All methods and seeds"), ("paired_deltas.png", "Paired benefit or harm against source EMA"),
                          ("pseudo_label_diagnostics.png", "Pseudo-label acceptance, correctness and class concentration"),
                          ("consistency_and_validation.png", "Consistency and source validation diagnostics"),
                          ("confusion_matrices.png", "Confusion matrices; repeated seed predictions are correlated"),
                          ("per_bearing_fault_recall.png", "Which faulty bearings are missed")]:
        lines.extend([f"![{caption}](../results/paderborn_commissioning_v1/figures/{name})", ""])
    lines.extend(["## How to interpret this experiment", "",
                  "A smaller training loss or more accepted pseudo-labels does not establish better diagnosis. Teacher and",
                  "student can become more consistent while reinforcing incorrect labels. Use the paired evaluation and",
                  "retrospective pseudo-label precision together. A high faulty-class share is not automatically collapse",
                  "because the adaptation collection is imbalanced; errors and bearing-level behavior must support that claim.", "",
                  "A matched source EMA removes much of the explanation due solely to extra source updates and weight averaging.",
                  "It does not isolate every component of the target objective: confidence masking, augmentation, loss weighting",
                  "and teacher feedback still require separate ablations. A positive pair is not proof of general superiority.", "",
                  "The source validation trajectories are logged diagnostics only. The fixed final epoch was used even if",
                  "validation worsened. A future stopping rule must be declared before new evaluation, not retrofitted to",
                  "pick the most favorable checkpoint from this study.", "",
                  "## Limits and verification", "",
                  "- This is binary source-access adaptation, not source-free adaptation, full CAT reproduction, or cross-machine validation.",
                  "- Baseline test results had already been inspected. Further development requires declared new versions and independent confirmation.",
                  "- Only six held-out bearings, including one healthy bearing, determine these results; windows do not increase specimen diversity.",
                  "- Nominal sampling rates and the original timestamp discrepancy remain documented in the baseline protocol.",
                  "- Target labels were blocked in training; the evaluator verifies frozen configuration, inputs, code and artifacts.",
                  "- Source index sequences and update budgets matched within each seed. See training_complete.json for provenance.",
                  "- A seed-7/low-speed replay checks exact tensors, histories and pseudo-label traces on the same machine.", "",
                  "## Evidence files", "",
                  "[All published evidence](../results/paderborn_commissioning_v1/) includes results_by_seed.csv, results_summary.csv,",
                  "paired_comparisons.csv, per_bearing_results.csv, training histories, pseudo-label traces, checkpoints and hash declarations.",
                  "See [the frozen protocol](PADERBORN_COMMISSIONING_PROTOCOL_V1.md) for settings and access boundaries.", ""])
    (ROOT / "reports/PADERBORN_COMMISSIONING_RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print("ANALYSIS COMPLETE: reconciled all 90 epoch traces; generated six figures and measured report.")
    print("Final-epoch accepted precision range:", analysis["accepted_precision_range"])


if __name__ == "__main__":
    main()
