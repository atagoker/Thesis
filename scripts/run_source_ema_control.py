"""Additional source-only EMA control, introduced after the first pilot.

Uses the same source checkpoints, source batch sequence, optimizer, epochs,
and EMA coefficient as the teacher method. It receives no target inputs or
labels. It saves the final EMA rather than selecting on target scores.
"""

import copy
import csv
import os
from pathlib import Path
import sys
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from thesislib.common import ROOT, BearingCNN, ema_update, read_json, set_seed, sha256, write_json
from run_pilot import source_data, source_loader, save_model


def main():
    destination = ROOT / "outputs/pilot_v1"
    cfg = read_json(destination / "config_snapshot.json")
    x, y = source_data("fit")
    device = torch.device("cuda")
    records = []
    for seed in cfg["seeds"]:
        set_seed(seed + 1000)
        folder = destination / f"seed_{seed}"
        student = BearingCNN().to(device)
        source = torch.load(folder / "source_best.pt", map_location="cpu", weights_only=True)
        student.load_state_dict(source["state_dict"])
        teacher = copy.deepcopy(student).eval().requires_grad_(False)
        optimizer = torch.optim.Adam(student.parameters(), lr=cfg["adaptation_learning_rate"], weight_decay=cfg["weight_decay"])
        loader = source_loader(x, y, cfg, seed + 1000)
        for _ in range(cfg["adaptation_epochs"]):
            student.train()
            for xb, yb in loader:
                loss = torch.nn.functional.cross_entropy(student(xb.to(device)), yb.to(device))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                ema_update(teacher, student, cfg["ema_alpha"])
        # Verify matched source sampling/optimization against the frozen control.
        original = torch.load(folder / "continued_source.pt", map_location="cpu", weights_only=True)
        max_difference = max(float((value.detach().cpu() - original["state_dict"][key]).abs().max())
                             for key, value in student.state_dict().items())
        if max_difference > 1e-6:
            raise RuntimeError(f"Source control replay differed by {max_difference}; investigate before reporting.")
        save_model(folder / "source_ema_control.pt", teacher, cfg, seed, "source_only_ema_control",
                   introduced_after_initial_pilot=True, true_target_inputs_used=False,
                   true_target_labels_used=False, selection_rule="fixed final epoch")
        records.append({"seed": seed, "student_replay_max_abs_difference": max_difference,
                        "checkpoint_sha256": sha256(folder / "source_ema_control.pt")})
        print(f"Source-only EMA control seed {seed}: saved; exact student replay difference={max_difference:g}", flush=True)
    write_json(destination / "source_ema_control_complete.json", {
        "reason": "Separate temporal averaging from unlabeled-target learning",
        "timing": "Added after inspection of the initial three-method pilot; no target-score-based hyperparameter tuning",
        "runs": records,
    })


if __name__ == "__main__":
    main()
