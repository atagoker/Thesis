"""Train source models and commission EMA teachers without target labels.

Target test data and all evaluator labels are intentionally absent from this
module. Run evaluate_pilot.py separately after all checkpoints are fixed.
"""

import argparse
import copy
import csv
from datetime import datetime, timezone
import os
from pathlib import Path
import sys
import time

# Required by deterministic CUDA matrix multiplication; set before torch import.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from thesislib.common import ROOT, CLASSES, BearingCNN, ema_update, masked_pseudo_loss, read_json, set_seed, sha256, write_json


def source_data(split):
    with np.load(ROOT / f"data/processed/pilot_v1/load0_{split}.npz") as data:
        return torch.from_numpy(data["x"].copy()), torch.from_numpy(data["y"].copy())


def target_inputs(load):
    with np.load(ROOT / f"data/processed/pilot_v1/load{load}_fit_unlabeled.npz") as data:
        if set(data.files) != {"x", "ids"}:
            raise ValueError("Target fit archive must not contain labels.")
        return torch.from_numpy(data["x"].copy())


def source_loader(x, y, cfg, seed):
    counts = torch.bincount(y, minlength=4)
    weights = (1 / counts.double())[y]
    generator = torch.Generator().manual_seed(seed)
    sampler = WeightedRandomSampler(weights, num_samples=len(y), replacement=True, generator=generator)
    return DataLoader(TensorDataset(x, y), batch_size=cfg["batch_size"], sampler=sampler,
                      num_workers=0, generator=torch.Generator().manual_seed(seed + 1))


@torch.no_grad()
def source_validation(model, x, y, device):
    model.eval()
    outputs = []
    for batch in x.split(256):
        outputs.append(model(batch.to(device)).argmax(1).cpu())
    predicted = torch.cat(outputs).numpy()
    return {"source_val_accuracy": float(accuracy_score(y.numpy(), predicted)),
            "source_val_macro_f1": float(f1_score(y.numpy(), predicted, labels=range(4), average="macro", zero_division=0))}


def save_model(path, model, cfg, seed, role, **metadata):
    torch.save({"state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                "config": cfg, "seed": seed, "role": role, "classes": CLASSES, **metadata}, path)


def write_history(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def train_source(cfg, seed, folder, device, x, y, val_x, val_y):
    set_seed(seed)
    model = BearingCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["source_learning_rate"], weight_decay=cfg["weight_decay"])
    loader = source_loader(x, y, cfg, seed)
    best_score, best_epoch, rows = -1, None, []
    started = time.perf_counter()
    for epoch in range(cfg["source_epochs"]):
        model.train()
        total, count = 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = nn.functional.cross_entropy(model(xb), yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(yb)
            count += len(yb)
        scores = source_validation(model, val_x, val_y, device)
        rows.append({"epoch": epoch + 1, "source_loss": total / count, **scores})
        if scores["source_val_macro_f1"] > best_score:
            best_score, best_epoch = scores["source_val_macro_f1"], epoch + 1
            save_model(folder / "source_best.pt", model, cfg, seed, "source_only", selected_epoch=best_epoch)
        print(f"seed {seed} source {epoch + 1:02d}/{cfg['source_epochs']} loss={total/count:.4f} source-val F1={scores['source_val_macro_f1']:.3f}", flush=True)
    write_history(folder / "source_history.csv", rows)
    return {"best_source_epoch": best_epoch, "best_source_validation_macro_f1": best_score,
            "source_training_seconds": time.perf_counter() - started,
            "source_updates_per_epoch": len(loader)}


def commission(cfg, seed, folder, device, x, y, target_load=None):
    # This function never receives or reads true target condition labels.
    set_seed(seed + 1000)
    checkpoint = torch.load(folder / "source_best.pt", map_location="cpu", weights_only=True)
    student = BearingCNN().to(device)
    student.load_state_dict(checkpoint["state_dict"])
    teacher = copy.deepcopy(student).eval()
    teacher.requires_grad_(False)
    optimizer = torch.optim.Adam(student.parameters(), lr=cfg["adaptation_learning_rate"], weight_decay=cfg["weight_decay"])
    source = source_loader(x, y, cfg, seed + 1000)
    if target_load is not None:
        target = DataLoader(TensorDataset(target_inputs(target_load)), batch_size=cfg["batch_size"],
                            shuffle=True, generator=torch.Generator().manual_seed(seed + 2000 + target_load))
    method = "continued_source" if target_load is None else f"teacher_target{target_load}"
    rows, started, updates = [], time.perf_counter(), 0
    for epoch in range(cfg["adaptation_epochs"]):
        student.train()
        totals = {"source_loss": 0.0, "target_loss": 0.0, "accepted": 0, "target_seen": 0,
                  "confidence_sum": 0.0, "accepted_class0": 0, "accepted_class1": 0,
                  "accepted_class2": 0, "accepted_class3": 0}
        target_iterator = iter(target) if target_load is not None else None
        for xb, yb in source:
            xb, yb = xb.to(device), yb.to(device)
            source_loss = nn.functional.cross_entropy(student(xb), yb)
            target_loss = source_loss.new_zeros(())
            if target_load is not None:
                try:
                    (xt,) = next(target_iterator)
                except StopIteration:
                    target_iterator = iter(target)
                    (xt,) = next(target_iterator)
                xt = xt.to(device)
                with torch.no_grad():
                    weak = xt + cfg["weak_noise_std"] * torch.randn_like(xt)
                    q = teacher(weak).softmax(1)
                    confidence, pseudo = q.max(1)
                    mask = confidence >= cfg["confidence_threshold"]
                gain = torch.empty((len(xt), 1, 1), device=device).uniform_(cfg["strong_gain_min"], cfg["strong_gain_max"])
                strong = xt * gain + cfg["strong_noise_std"] * torch.randn_like(xt)
                target_loss = masked_pseudo_loss(student(strong), pseudo, mask)
                totals["accepted"] += int(mask.sum())
                totals["target_seen"] += len(xt)
                totals["confidence_sum"] += float(confidence.sum())
                for class_id in range(4):
                    totals[f"accepted_class{class_id}"] += int(((pseudo == class_id) & mask).sum())
            ramp = (epoch + 1) / cfg["adaptation_epochs"]
            loss = source_loss + cfg["target_loss_weight"] * ramp * target_loss
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite adaptation objective.")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if target_load is not None:
                ema_update(teacher, student, cfg["ema_alpha"])
            updates += 1
            totals["source_loss"] += source_loss.item()
            totals["target_loss"] += target_loss.item()
        denominator = max(1, totals["target_seen"])
        rows.append({"epoch": epoch + 1, "source_loss": totals["source_loss"] / len(source),
                     "target_loss": totals["target_loss"] / len(source),
                     "coverage": totals["accepted"] / denominator,
                     "mean_confidence": totals["confidence_sum"] / denominator,
                     **{f"accepted_class{k}": totals[f"accepted_class{k}"] for k in range(4)}})
        print(f"seed {seed} {method} {epoch+1:02d}/{cfg['adaptation_epochs']} coverage={rows[-1]['coverage']:.3f}", flush=True)
    # Fixed final epoch. No target-label-based selection.
    final_model = student if target_load is None else teacher
    save_model(folder / f"{method}.pt", final_model, cfg, seed, method,
               selected_epoch=cfg["adaptation_epochs"], updates=updates,
               selection_rule="fixed final epoch", true_target_labels_used=False)
    if target_load is not None:
        save_model(folder / f"student_target{target_load}.pt", student, cfg, seed,
                   "student_diagnostic_only", selection_rule="fixed final epoch")
    write_history(folder / f"{method}_history.csv", rows)
    return {"method": method, "seconds": time.perf_counter() - started,
            "updates": updates, "final_coverage": rows[-1]["coverage"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/pilot_v1.json")
    args = parser.parse_args()
    cfg = read_json(args.config)
    prepared = read_json(ROOT / "data/manifests/pilot_v1_summary.json")
    if prepared["config_sha256"] != sha256(args.config):
        raise ValueError("Protocol changed since preparation. Prepare again before training.")
    if not torch.cuda.is_available():
        raise RuntimeError("This recorded pilot configuration requires the verified GPU.")
    device = torch.device("cuda")
    destination = ROOT / "outputs/pilot_v1"
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / "training_complete.json").exists():
        raise RuntimeError("Completed pilot exists; preserve it and use a new protocol/output version for a new experiment.")
    write_json(destination / "config_snapshot.json", cfg)
    x, y = source_data("fit")
    vx, vy = source_data("validation")
    all_runs = []
    print(f"GPU: {torch.cuda.get_device_name(0)}; source fit={len(y)}, source validation={len(vy)}", flush=True)
    for seed in cfg["seeds"]:
        folder = destination / f"seed_{seed}"
        folder.mkdir(exist_ok=True)
        info = train_source(cfg, seed, folder, device, x, y, vx, vy)
        info["seed"] = seed
        info["adaptations"] = [commission(cfg, seed, folder, device, x, y)]
        for load in cfg["target_loads"]:
            info["adaptations"].append(commission(cfg, seed, folder, device, x, y, target_load=load))
        all_runs.append(info)
    checkpoints = {str(p.relative_to(destination)): sha256(p) for p in destination.glob("seed_*/*.pt")}
    write_json(destination / "training_complete.json", {
        "completed_utc": datetime.now(timezone.utc).isoformat(), "config_sha256": sha256(args.config),
        "runs": all_runs, "checkpoint_sha256": checkpoints,
        "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
        "parameter_count": sum(p.numel() for p in BearingCNN().parameters()),
        "target_labels_used_for_training_or_selection": False,
        "target_evaluation_performed": False,
    })
    print("Training complete. All checkpoints fixed before separate target evaluation.", flush=True)


if __name__ == "__main__":
    main()
