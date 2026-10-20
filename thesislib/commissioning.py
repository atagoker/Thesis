"""Label-separated, source-access teacher-student training for Paderborn.

This new module deliberately leaves the frozen baseline implementation intact.
"""

import copy
import csv
import hashlib
import os
from pathlib import Path
import time

import numpy as np
from sklearn.metrics import f1_score
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from thesislib.common import BearingCNN, ROOT, ema_update, masked_pseudo_loss, read_json, set_seed, sha256
from thesislib.paderborn_ml import aggregate_recordings, probabilities

CONFIG_PATH = ROOT / "configs/paderborn_commissioning_v1.json"
PROTOCOL_PATH = ROOT / "reports/PADERBORN_COMMISSIONING_PROTOCOL_V1.md"
IMPLEMENTATION_PATHS = [
    "thesislib/commissioning.py", "thesislib/common.py", "thesislib/paderborn_ml.py",
    "scripts/train_paderborn_commissioning.py",
]


class TrainingAccessGuard:
    """Scoped accidental-access protection; not a general security sandbox."""

    def __init__(self):
        self.enabled = False
        self.target_condition = None

    def __call__(self, event, args):
        if not self.enabled or event != "open" or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(args[0])).resolve().as_posix().lower()
        mode = args[1]
        # Output writes are allowed; all read rules apply to read/write modes too.
        if isinstance(mode, str) and "r" not in mode and "+" not in mode and any(c in mode for c in "wax"):
            return
        relative = path.removeprefix(ROOT.resolve().as_posix().lower() + "/")
        denied = (relative.startswith(("data/raw/", "results/", "reports/"))
                  or "paderborn_v1_records.csv" in path
                  or "/evaluator/" in path or "_labels.npz" in path
                  or "/predictions/" in path or relative.endswith("evaluation.json")
                  or "results_by_seed.csv" in path or "results_summary.csv" in path)
        if "/paderborn_v1/" in path:
            name = Path(path).name
            allowed = {"source_fit.npz", "source_validation.npz"}
            if self.target_condition is not None:
                allowed.add(f"target_adaptation_{self.target_condition.lower()}_unlabeled.npz")
            denied = denied or name not in allowed
        if denied:
            raise PermissionError("Commissioning trainer cannot read evaluator/test/curator data or unauthorized target inputs.")


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def portable_path(path):
    return Path(str(path).replace("\\", "/"))


def validate_provenance(cfg):
    baseline = read_json(ROOT / cfg["baseline_config"])
    frozen = read_json(ROOT / cfg["baseline_completion"])
    prepared = read_json(ROOT / cfg["prepared_summary"])
    if sha256(ROOT / cfg["baseline_config"]) != frozen["config_sha256"]:
        raise ValueError("Baseline configuration changed.")
    if sha256(ROOT / cfg["prepared_summary"]) != frozen["prepared_summary_sha256"]:
        raise ValueError("Prepared data declaration changed.")
    if prepared["config_sha256"] != frozen["config_sha256"]:
        raise ValueError("Prepared data and baseline protocol disagree.")
    for relative, digest in frozen["implementation_sha256"].items():
        if sha256(ROOT / portable_path(relative)) != digest:
            raise ValueError(f"Frozen baseline implementation changed: {relative}")
    for relative, digest in frozen["checkpoint_sha256"].items():
        if sha256((ROOT / cfg["baseline_completion"]).parent / portable_path(relative)) != digest:
            raise ValueError(f"Frozen baseline checkpoint changed: {relative}")
    for key in ["classes", "seeds", "target_conditions"]:
        if cfg[key] != baseline[key]:
            raise ValueError(f"Commissioning changed frozen {key}.")
    if cfg["target_labels_used_for_training_or_selection"] or not cfg["source_data_available_during_adaptation"]:
        raise ValueError("Unexpected label/source access protocol.")
    if not 0 <= cfg["ema_alpha"] < 1 or not .5 <= cfg["confidence_threshold"] <= 1:
        raise ValueError("Invalid EMA or confidence threshold.")
    if cfg["adaptation_epochs"] < 1 or cfg["target_loss_weight"] < 0:
        raise ValueError("Invalid adaptation budget/weight.")
    return baseline, frozen, prepared


def load_archive(cfg, prepared, name, labeled):
    path = ROOT / cfg["prepared_directory"] / f"{name}.npz"
    if sha256(path) != prepared["prepared_sha256"][path.name]:
        raise ValueError(f"Prepared input changed: {path}")
    with np.load(path, allow_pickle=False) as data:
        expected = {"x", "y", "ids"} if labeled else {"x", "ids"}
        if set(data.files) != expected:
            raise ValueError(f"Incorrect label boundary/schema: {path}")
        x, ids = data["x"].copy(), data["ids"].copy()
        if x.ndim != 3 or x.shape[1:] != (1, 2048) or not np.isfinite(x).all() or len(x) != len(ids):
            raise ValueError("Invalid prepared input geometry or values.")
        if len(np.unique(ids)) != len(ids):
            raise ValueError("Duplicate window IDs.")
        y = data["y"].copy() if labeled else None
        if labeled and (len(y) != len(x) or set(np.unique(y)) != {0, 1}):
            raise ValueError("Source labels must have both binary classes.")
    return torch.from_numpy(x), y, ids


def source_loader(x, y, batch_size, seed):
    labels = torch.as_tensor(y, dtype=torch.long)
    counts = torch.bincount(labels, minlength=2)
    if torch.any(counts == 0):
        raise ValueError("Both source classes are required.")
    sampler = WeightedRandomSampler((1.0 / counts.double())[labels], len(y), replacement=True,
                                    generator=torch.Generator().manual_seed(seed + 1000))
    return DataLoader(TensorDataset(x, labels, torch.arange(len(x))), sampler=sampler,
                      batch_size=batch_size, num_workers=0,
                      generator=torch.Generator().manual_seed(seed + 1001))


def make_teacher(student):
    return copy.deepcopy(student).eval().requires_grad_(False)


def views(x, cfg, generator):
    weak = x + cfg["weak_noise_std"] * torch.randn(x.shape, device=x.device, generator=generator)
    gain = torch.empty((len(x), 1, 1), device=x.device).uniform_(
        cfg["strong_gain_min"], cfg["strong_gain_max"], generator=generator)
    strong = gain * x + cfg["strong_noise_std"] * torch.randn(x.shape, device=x.device, generator=generator)
    return weak, strong


def source_score(model, validation, device):
    x, y, ids = validation
    _, p, truth = aggregate_recordings(probabilities(model, x, device), ids, y)
    return float(f1_score(truth, p[:, 1] >= .5, labels=[0, 1], average="macro", zero_division=0))


def cpu_state(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def train_run(cfg, seed, initial_state, source, validation, target=None, target_index=0, verbose=True):
    """The only target argument is an input tensor; no target labels/IDs enter."""
    set_seed(seed + 1000)
    device = initial_state[next(iter(initial_state))].device
    student = BearingCNN(num_classes=2).to(device)
    student.load_state_dict(initial_state)
    teacher = make_teacher(student)
    optimizer = torch.optim.Adam(student.parameters(), lr=cfg["adaptation_learning_rate"],
                                 weight_decay=cfg["weight_decay"])
    sx, sy = source
    loader = source_loader(sx, sy, cfg["batch_size"], seed)
    if target is not None:
        target_loader = DataLoader(TensorDataset(target, torch.arange(len(target))),
                                  batch_size=cfg["batch_size"], shuffle=True, num_workers=0,
                                  generator=torch.Generator().manual_seed(seed + 2000 + target_index))
        augmentation_rng = torch.Generator(device=device).manual_seed(seed + 3000 + target_index)
    history, trace_parts = [], []
    sampled = hashlib.sha256()
    started = time.perf_counter()
    for epoch in range(1, cfg["adaptation_epochs"] + 1):
        student.train()
        totals = dict(source_loss=0., source_seen=0, target_loss=0., seen=0,
                      accepted=0, confidence=0., disagreement=0, accepted_disagreement=0,
                      pseudo_healthy=0, pseudo_faulty=0, accepted_healthy=0, accepted_faulty=0)
        iterator = iter(target_loader) if target is not None else None
        for xb, yb, indices in loader:
            sampled.update(indices.numpy().astype("<i8").tobytes())
            xb, yb = xb.to(device), yb.to(device)
            source_loss = nn.functional.cross_entropy(student(xb), yb)
            target_loss = source_loss.new_zeros(())
            if target is not None:
                try:
                    xt, target_indices = next(iterator)
                except StopIteration:
                    iterator = iter(target_loader)
                    xt, target_indices = next(iterator)
                xt = xt.to(device)
                weak, strong = views(xt, cfg, augmentation_rng)
                with torch.no_grad():
                    confidence, pseudo = teacher(weak).softmax(1).max(1)
                    mask = confidence >= cfg["confidence_threshold"]
                logits = student(strong)
                target_loss = masked_pseudo_loss(logits, pseudo, mask)
                predicted = logits.detach().argmax(1)
                totals["seen"] += len(xt)
                totals["accepted"] += int(mask.sum())
                totals["confidence"] += float(confidence.sum())
                totals["disagreement"] += int((predicted != pseudo).sum())
                totals["accepted_disagreement"] += int(((predicted != pseudo) & mask).sum())
                for k, label in enumerate(["healthy", "faulty"]):
                    totals[f"pseudo_{label}"] += int((pseudo == k).sum())
                    totals[f"accepted_{label}"] += int(((pseudo == k) & mask).sum())
                trace_parts.append({"epoch": np.full(len(xt), epoch, dtype=np.int16),
                                    "index": target_indices.numpy().astype(np.int32),
                                    "pseudo": pseudo.cpu().numpy().astype(np.int8),
                                    "confidence": confidence.cpu().numpy(),
                                    "accepted": mask.cpu().numpy(),
                                    "student_prediction": predicted.cpu().numpy().astype(np.int8)})
                totals["target_loss"] += float(target_loss.detach()) * len(xt)
            weight = cfg["target_loss_weight"] * epoch / cfg["adaptation_epochs"]
            loss = source_loss + weight * target_loss
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite commissioning objective.")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            ema_update(teacher, student, cfg["ema_alpha"])
            totals["source_loss"] += float(source_loss.detach()) * len(yb)
            totals["source_seen"] += len(yb)
        seen = max(totals["seen"], 1)
        row = {"epoch": epoch, "updates": epoch * len(loader),
               "source_loss": totals["source_loss"] / totals["source_seen"],
               "target_loss_full_batch": totals["target_loss"] / seen,
               "target_weight": weight if target is not None else 0.,
               "target_exposures": totals["seen"], "accepted": totals["accepted"],
               "acceptance": totals["accepted"] / seen,
               "mean_confidence": totals["confidence"] / seen,
               "teacher_student_disagreement": totals["disagreement"] / seen,
               "accepted_disagreement": (totals["accepted_disagreement"] / totals["accepted"]
                                         if totals["accepted"] else None),
               **{key: totals[key] for key in ["pseudo_healthy", "pseudo_faulty", "accepted_healthy", "accepted_faulty"]},
               "source_val_student_macro_f1": source_score(student, validation, device),
               "source_val_teacher_macro_f1": source_score(teacher, validation, device)}
        history.append(row)
        if verbose:
            print(f"seed={seed} {'control' if target is None else 'target'+str(target_index)} "
                  f"epoch={epoch}/{cfg['adaptation_epochs']} coverage={row['acceptance']:.3f} "
                  f"source-val teacher F1={row['source_val_teacher_macro_f1']:.3f}", flush=True)
    trace = {key: np.concatenate([part[key] for part in trace_parts]) for key in trace_parts[0]} if trace_parts else {}
    return {"student": cpu_state(student), "teacher": cpu_state(teacher), "history": history, "trace": trace,
            "seconds": time.perf_counter() - started, "source_sequence_sha256": sampled.hexdigest(),
            "updates": len(loader) * cfg["adaptation_epochs"]}
