"""Train three source-only binary baselines; never open target/evaluator arrays."""

from datetime import datetime, timezone
import argparse
import csv
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
from sklearn.metrics import f1_score
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from thesislib.common import ROOT, BearingCNN, read_json, set_seed, sha256, write_json
from thesislib.paderborn import validate_protocol
from thesislib.paderborn_ml import aggregate_recordings, probabilities


def deny_target_access(event, args):
    if event != "open" or not isinstance(args[0], (str, bytes, os.PathLike)):
        return
    path = os.fsdecode(args[0]).replace("\\", "/").lower()
    if ("paderborn_v1/evaluator/" in path or "paderborn_v1_records.csv" in path
            or ("paderborn_v1/" in path and
                ("target_adaptation_" in path or "evaluation_" in path))):
        raise PermissionError("Source-only training must not open target/evaluation data or curator labels.")


def load_source(name, prepared):
    path = ROOT / f"data/processed/paderborn_v1/{name}.npz"
    if sha256(path) != prepared["prepared_sha256"][path.name]:
        raise ValueError(f"Prepared source archive changed: {path}")
    with np.load(path, allow_pickle=False) as data:
        if set(data.files) != {"x", "y", "ids"}:
            raise ValueError("Unexpected source archive schema.")
        return torch.from_numpy(data["x"].copy()), data["y"].copy(), data["ids"].copy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-seed", type=int, choices=[7, 17, 27],
                        help="Retrain one fixed seed in a separate folder and compare exact checkpoint tensors.")
    args = parser.parse_args()
    sys.addaudithook(deny_target_access)
    config_path = ROOT / "configs/paderborn_baseline_v1.json"
    cfg = read_json(config_path)
    validate_protocol(cfg)
    prepared = read_json(ROOT / "data/manifests/paderborn_v1_summary.json")
    if prepared["config_sha256"] != sha256(config_path):
        raise ValueError("Configuration changed after preparation.")
    original_out = ROOT / "outputs/paderborn_baseline_v1"
    out = original_out if args.replay_seed is None else original_out / f"verification_replay_{args.replay_seed}"
    if args.replay_seed is not None and not (original_out / "training_complete.json").exists():
        raise RuntimeError("Train the original baseline before requesting a replay.")
    out.mkdir(parents=True, exist_ok=True)
    if (out / "training_complete.json").exists():
        raise RuntimeError("Completed baseline exists. Preserve it; declare a new protocol to retrain.")
    if not torch.cuda.is_available():
        raise RuntimeError("This recorded baseline requires the verified CUDA GPU.")
    device = torch.device("cuda")
    x, y, _ = load_source("source_fit", prepared)
    vx, vy, vids = load_source("source_validation", prepared)
    runs = []
    write_json(out / "config_snapshot.json", cfg)
    print(f"GPU: {torch.cuda.get_device_name(0)}; source windows: {len(y)}; validation: {len(vy)}", flush=True)
    for seed in cfg["seeds"] if args.replay_seed is None else [args.replay_seed]:
        set_seed(seed)
        model = BearingCNN(num_classes=2).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
        labels = torch.from_numpy(y)
        counts = torch.bincount(labels, minlength=2)
        sampler = WeightedRandomSampler((1 / counts.double())[labels], len(y), replacement=True,
                                        generator=torch.Generator().manual_seed(seed))
        loader = DataLoader(TensorDataset(x, labels), sampler=sampler, batch_size=cfg["batch_size"],
                            num_workers=0, generator=torch.Generator().manual_seed(seed + 1))
        folder = out / f"seed_{seed}"
        folder.mkdir(exist_ok=True)
        history = []
        best, best_epoch = -1, None
        started = time.perf_counter()
        for epoch in range(1, cfg["source_epochs"] + 1):
            model.train()
            total = 0.
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                loss = nn.functional.cross_entropy(model(xb), yb)
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite source objective.")
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total += loss.item() * len(yb)
            _, rp, ry = aggregate_recordings(probabilities(model, vx, device), vids, vy)
            score = float(f1_score(ry, (rp[:, 1] >= .5).astype(int), labels=[0, 1], average="macro", zero_division=0))
            history.append({"epoch": epoch, "source_loss": total / len(y), "source_val_recording_macro_f1": score})
            if score > best:
                best, best_epoch = score, epoch
                torch.save({"state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                            "classes": cfg["classes"], "config": cfg, "seed": seed,
                            "role": "paderborn_binary_source_only", "selected_epoch": epoch,
                            "source_validation_recording_macro_f1": score,
                            "target_labels_used": False}, folder / "source_best.pt")
            print(f"seed {seed} epoch {epoch:02d}/{cfg['source_epochs']} loss={total/len(y):.4f} "
                  f"source-val recording F1={score:.4f}", flush=True)
        with (folder / "source_history.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
        runs.append({"seed": seed, "selected_epoch": best_epoch, "source_val_macro_f1": best,
                     "training_seconds": time.perf_counter() - started, "updates": len(loader) * cfg["source_epochs"]})
    sources = ["thesislib/common.py", "thesislib/paderborn.py", "thesislib/paderborn_ml.py",
               "scripts/prepare_paderborn.py", "scripts/train_paderborn_baseline.py"]
    write_json(out / "training_complete.json", {
        "completed_utc": datetime.now(timezone.utc).isoformat(), "protocol_id": cfg["protocol_id"],
        "config_sha256": sha256(config_path), "prepared_summary_sha256": sha256(ROOT / "data/manifests/paderborn_v1_summary.json"),
        "checkpoint_sha256": {str(p.relative_to(out)): sha256(p) for p in out.glob("seed_*/*.pt")},
        "implementation_sha256": {p: sha256(ROOT / p) for p in sources}, "runs": runs,
        "torch": str(torch.__version__), "gpu": torch.cuda.get_device_name(0),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "target_inputs_used": False, "target_labels_used_for_training_or_selection": False,
        "audit_hook_blocks_target_and_evaluator_file_access": True,
        "target_evaluation_performed": False})
    if args.replay_seed is not None:
        seed = args.replay_seed
        reference = torch.load(original_out / f"seed_{seed}/source_best.pt", map_location="cpu", weights_only=True)
        replay = torch.load(out / f"seed_{seed}/source_best.pt", map_location="cpu", weights_only=True)
        differences = [float((reference["state_dict"][k] - replay["state_dict"][k]).abs().max())
                       for k in reference["state_dict"]]
        exact = all(torch.equal(reference["state_dict"][k], replay["state_dict"][k]) for k in reference["state_dict"])
        evidence = {"seed": seed, "exact_checkpoint_tensor_match": exact,
                    "maximum_absolute_parameter_difference": max(differences),
                    "selected_epoch_matches": reference["selected_epoch"] == replay["selected_epoch"],
                    "scope": "Same installed software, GPU, data and configuration; not a cross-device guarantee.",
                    "audit_hook_blocks_target_and_evaluator_file_access": True}
        write_json(original_out / f"reproducibility_seed_{seed}.json", evidence)
        if not exact or not evidence["selected_epoch_matches"]:
            raise RuntimeError("Deterministic replay differed; inspect reproducibility evidence.")
        print("REPLAY PASSED: exact checkpoint tensors match.", flush=True)
    else:
        print("BASELINE TRAINING COMPLETE. Checkpoints frozen before separate evaluation.", flush=True)


if __name__ == "__main__":
    main()
