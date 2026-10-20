"""Replay one adaptation while blocking all evaluator-label access."""

import os
from pathlib import Path
import shutil
import sys
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from thesislib.common import ROOT, read_json, write_json
from run_pilot import commission, source_data


def main():
    destination = ROOT / "outputs/pilot_v1"
    cfg = read_json(destination / "config_snapshot.json")
    seed, load = cfg["seeds"][0], cfg["target_loads"][0]
    original_folder = destination / f"seed_{seed}"
    replay = destination / "verification_replay" / f"seed_{seed}"
    replay.mkdir(parents=True, exist_ok=True)
    shutil.copy2(original_folder / "source_best.pt", replay / "source_best.pt")
    x, y = source_data("fit")
    opened_paths = []

    def audit(event, args):
        if event != "open" or not args or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        value = os.fsdecode(args[0]).replace("\\", "/").lower()
        if "/evaluator/" in value or value.endswith("_labels.npz"):
            raise RuntimeError(f"Forbidden target ground-truth access during commissioning: {value}")
        if "data/processed/pilot_v1/" in value:
            opened_paths.append(value)

    sys.addaudithook(audit)
    commission(cfg, seed, replay, torch.device("cuda"), x, y, target_load=load)
    first = torch.load(original_folder / f"teacher_target{load}.pt", weights_only=True, map_location="cpu")
    second = torch.load(replay / f"teacher_target{load}.pt", weights_only=True, map_location="cpu")
    maximum = max(float((first["state_dict"][key] - value).abs().max())
                  for key, value in second["state_dict"].items())
    if maximum != 0:
        raise RuntimeError(f"Deterministic teacher replay differed: {maximum}")
    write_json(destination / "reproducibility_check.json", {
        "status": "passed", "seed": seed, "target_load_hp": load,
        "evaluator_label_access": "blocked by Python audit hook during replay",
        "training_facing_data_paths_opened": sorted(set(opened_paths)),
        "max_abs_teacher_parameter_difference": maximum,
        "scope": "Exact replay on this software/hardware; does not guarantee cross-device determinism",
    })
    print("Replay passed: identical teacher weights with evaluator-label access blocked.", flush=True)


if __name__ == "__main__":
    main()
