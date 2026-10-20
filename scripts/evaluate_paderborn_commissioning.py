"""Evaluate all frozen methods; never select a new model or decision threshold."""

from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from thesislib.common import BearingCNN, sha256, write_json
from thesislib.commissioning import write_csv
from thesislib.commissioning_evaluation import evaluation_arrays, metrics, model_path, verify_frozen_run
from thesislib.paderborn_ml import aggregate_recordings, probabilities

METRIC_NAMES = ["accuracy", "balanced_accuracy", "macro_f1", "fault_recall", "healthy_false_positive_rate",
                "roc_auc", "fault_average_precision"]


def main():
    cfg, baseline, prepared, out, frozen = verify_frozen_run()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(4)
    rows, bearing_rows = [], []
    predictions = out / "predictions"
    predictions.mkdir(exist_ok=True)
    arrays = {condition: evaluation_arrays(cfg, prepared, f"evaluation_{condition}")
              for condition in [baseline["source_condition"], *cfg["target_conditions"]]}
    for target in cfg["target_conditions"]:
        for seed in cfg["seeds"]:
            for method in cfg["methods"]:
                path = model_path(cfg, out, seed, target, method)
                checkpoint = torch.load(path, map_location="cpu", weights_only=True)
                model = BearingCNN(num_classes=2).to(device)
                model.load_state_dict(checkpoint["state_dict"])
                for scope, condition in [("target", target), ("source_retention", baseline["source_condition"])]:
                    x, ids, y = arrays[condition]
                    p = probabilities(model, x, device)
                    records, rp, ry = aggregate_recordings(p, ids, y)
                    common = {"scope": scope, "adaptation_target": target, "evaluation_condition": condition,
                              "seed": seed, "method": method}
                    for level, truth, prob in [("window", y, p), ("recording", ry, rp)]:
                        rows.append({**common, "level": level, **metrics(truth, prob)})
                    np.savez_compressed(predictions / f"{scope}_{target}_{method}_seed{seed}.npz",
                                        ids=records, y=ry, probabilities=rp)
                    bearings = np.asarray([str(record).split("_")[3] for record in records])
                    if set(bearings) != set(baseline["bearing_groups"]["evaluation"]):
                        raise ValueError("Unexpected test bearing group.")
                    for bearing in sorted(set(bearings)):
                        mask = bearings == bearing
                        predicted = rp[mask, 1] >= .5
                        bearing_rows.append({**common, "bearing": bearing, "class_id": int(ry[mask][0]),
                                             "recordings": int(mask.sum()),
                                             "mean_fault_score": float(rp[mask, 1].mean()),
                                             "accuracy": float((predicted == ry[mask]).mean()),
                                             "predicted_faulty": int(predicted.sum())})
                print(f"{target} seed={seed} {method}: target recording F1={rows[-3]['macro_f1']:.4f}", flush=True)
    write_csv(out / "results_by_seed.csv", rows)
    write_csv(out / "per_bearing_results.csv", bearing_rows)
    summary = []
    for scope in ["target", "source_retention"]:
        for target in cfg["target_conditions"]:
            for method in cfg["methods"]:
                selected = [row for row in rows if row["scope"] == scope and row["adaptation_target"] == target
                            and row["method"] == method and row["level"] == "recording"]
                assert len(selected) == 3
                item = {"scope": scope, "adaptation_target": target, "method": method,
                        "recordings_per_seed": selected[0]["n"], "seeds": len(selected)}
                for name in METRIC_NAMES:
                    values = [row[name] for row in selected]
                    item[f"{name}_mean"] = float(np.mean(values))
                    item[f"{name}_sd"] = float(np.std(values, ddof=1))
                summary.append(item)
    write_csv(out / "results_summary.csv", summary)
    pairs = []
    index = {(row["adaptation_target"], row["seed"], row["method"]): row for row in rows
             if row["scope"] == "target" and row["level"] == "recording"}
    for target in cfg["target_conditions"]:
        for seed in cfg["seeds"]:
            teacher = index[(target, seed, "teacher_student")]
            for control in ["source_only", "continued_source", "source_ema"]:
                reference = index[(target, seed, control)]
                pairs.append({"target": target, "seed": seed, "control": control,
                              "teacher_macro_f1": teacher["macro_f1"], "control_macro_f1": reference["macro_f1"],
                              "delta_macro_f1_pp": 100 * (teacher["macro_f1"] - reference["macro_f1"]),
                              "negative_transfer": teacher["macro_f1"] < reference["macro_f1"] - 1e-12,
                              "delta_fault_recall_pp": 100 * (teacher["fault_recall"] - reference["fault_recall"]),
                              "delta_healthy_fpr_pp": 100 * (teacher["healthy_false_positive_rate"] - reference["healthy_false_positive_rate"])})
    write_csv(out / "paired_comparisons.csv", pairs)
    comparisons = []
    for control in ["source_only", "continued_source", "source_ema"]:
        selected = [row for row in pairs if row["control"] == control]
        deltas = [row["delta_macro_f1_pp"] for row in selected]
        comparisons.append({"control": control, "paired_task_seeds": len(selected),
                            "mean_delta_macro_f1_pp": float(np.mean(deltas)),
                            "improved": sum(d > 1e-10 for d in deltas), "worse": sum(d < -1e-10 for d in deltas),
                            "tied": sum(abs(d) <= 1e-10 for d in deltas)})
    evidence = {"evaluated_utc": datetime.now(timezone.utc).isoformat(),
                "training_completion_sha256": sha256(out / "training_complete.json"),
                "evaluator_sha256": {"scripts/evaluate_paderborn_commissioning.py": sha256(Path(__file__)),
                                     "thesislib/commissioning_evaluation.py": sha256(Path(__file__).resolve().parents[1] / "thesislib/commissioning_evaluation.py")},
                "all_training_artifacts_verified": True,
                "target_labels_used_for_training_or_selection": False,
                "threshold": .5, "primary_level": "recording", "summary": summary, "comparisons": comparisons,
                "overall_target_macro_f1": {method: float(np.mean([r["macro_f1"] for r in index.values()
                                                                                if r["method"] == method])) for method in cfg["methods"]},
                "notes": ["Exploratory fixed-split evidence after baseline inspection.",
                          "Seed SD is not uncertainty over independent bearings or machines.",
                          "Source-retention control results repeat across target columns; do not treat as independent.",
                          "A single healthy bearing is represented in the held-out group."]}
    write_json(out / "evaluation.json", evidence)
    print("PAIRED COMPARISONS:", comparisons, flush=True)


if __name__ == "__main__":
    main()
