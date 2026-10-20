"""Offline-only reference-label access and metrics for frozen commissioning runs."""

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, roc_auc_score)
import torch

from thesislib.common import ROOT, read_json, sha256
from thesislib.commissioning import CONFIG_PATH, PROTOCOL_PATH, portable_path, validate_provenance


def verify_frozen_run():
    cfg = read_json(CONFIG_PATH)
    baseline, _, prepared = validate_provenance(cfg)
    out = ROOT / cfg["output_directory"]
    frozen = read_json(out / "training_complete.json")
    expected = {"config_sha256": sha256(CONFIG_PATH), "protocol_sha256": sha256(PROTOCOL_PATH),
                "prepared_summary_sha256": sha256(ROOT / cfg["prepared_summary"]),
                "baseline_completion_sha256": sha256(ROOT / cfg["baseline_completion"])}
    for key, value in expected.items():
        if frozen[key] != value:
            raise ValueError(f"Frozen provenance changed: {key}")
    if frozen["replay"] or len(frozen["runs"]) != 12 or len(frozen["checkpoint_sha256"]) != 24:
        raise ValueError("Expected all 3 source controls and 9 target adaptations.")
    for relative, digest in frozen["implementation_sha256"].items():
        if sha256(ROOT / portable_path(relative)) != digest:
            raise ValueError(f"Frozen training code changed: {relative}")
    for relative, digest in frozen["artifact_sha256"].items():
        if sha256(out / portable_path(relative)) != digest:
            raise ValueError(f"Frozen training artifact changed: {relative}")
    for seed in cfg["seeds"]:
        runs = [row for row in frozen["runs"] if row["seed"] == seed]
        if len({(r["updates"], r["source_sequence_sha256"]) for r in runs}) != 1:
            raise ValueError("Unmatched source update sequence/budget.")
    return cfg, baseline, prepared, out, frozen


def evaluation_arrays(cfg, prepared, key):
    directory = ROOT / cfg["prepared_directory"]
    input_path = directory / f"{key}_unlabeled.npz"
    label_path = directory / "evaluator" / f"{key}_labels.npz"
    digests = {k.replace("\\", "/"): value for k, value in prepared["prepared_sha256"].items()}
    for path in [input_path, label_path]:
        if sha256(path) != digests[path.relative_to(directory).as_posix()]:
            raise ValueError(f"Changed evaluation input/labels: {path}")
    with np.load(input_path, allow_pickle=False) as data, np.load(label_path, allow_pickle=False) as labels:
        if set(data.files) != {"x", "ids"} or set(labels.files) != {"y", "ids"}:
            raise ValueError("Invalid evaluator schema.")
        if not np.array_equal(data["ids"], labels["ids"]) or len(data["x"]) != len(labels["y"]):
            raise ValueError("Evaluator labels do not align with input IDs.")
        return torch.from_numpy(data["x"].copy()), data["ids"].copy(), labels["y"].copy()


def metrics(y, p):
    pred = p[:, 1] >= .5
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"n": len(y), "accuracy": float(accuracy_score(y, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, labels=[0, 1], average="macro", zero_division=0)),
            "fault_recall": float(tp / (tp + fn)), "healthy_false_positive_rate": float(fp / (tn + fp)),
            "roc_auc": float(roc_auc_score(y, p[:, 1])),
            "fault_average_precision": float(average_precision_score(y, p[:, 1])),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def model_path(cfg, out, seed, target, method):
    if method == "source_only":
        return (ROOT / cfg["baseline_completion"]).parent / f"seed_{seed}/source_best.pt"
    group = target if method == "teacher_student" else "source_control"
    return out / f"seed_{seed}" / group / f"{method}.pt"
