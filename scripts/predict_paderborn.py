"""Run a prototype binary prediction on one Paderborn-format MATLAB recording."""

import argparse
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from thesislib.common import ROOT, BearingCNN, sha256, write_json
from thesislib.paderborn import make_windows, read_vibration
from thesislib.paderborn_ml import probabilities


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path,
                        default=ROOT / "outputs/paderborn_baseline_v1/seed_7/source_best.pt")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/paderborn_demo/prediction.json")
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint["role"] != "paderborn_binary_source_only" or checkpoint["classes"] != ["healthy", "faulty"]:
        raise ValueError("Expected the two-class Paderborn baseline checkpoint, not a CWRU model.")
    raw, quality = read_vibration(args.file)
    x = torch.from_numpy(make_windows(raw, checkpoint["config"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(4)
    model = BearingCNN(num_classes=2).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    window_probabilities = probabilities(model, x, device)
    mean = window_probabilities.mean(axis=0)
    decision = int(mean[1] >= .5)
    result = {"input_file": str(args.file.resolve()), "input_sha256": sha256(args.file),
              "checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": sha256(args.checkpoint),
              "model_role": checkpoint["role"], "device": str(device), "vibration_channel": "vibration_1",
              "window_count": len(x), "predicted_class": checkpoint["classes"][decision],
              "mean_probabilities": {label: float(value) for label, value in zip(checkpoint["classes"], mean)},
              "window_prediction_counts": {label: int(((window_probabilities[:, 1] >= .5) == i).sum())
                                           for i, label in enumerate(checkpoint["classes"])},
              "input_quality": quality, "threshold": .5,
              "warning": "Prototype score, not a calibrated failure probability. No unknown-fault rejection. "
                         "Running a known test recording is a demonstration, not a new independent validation."}
    write_json(args.output, result)
    print(f"Predicted recording class: {result['predicted_class']}")
    print(f"Mean model score for faulty: {mean[1]:.4f} (not a calibrated risk estimate)")
    print(f"Windows: {len(x)}; device: {device}; report: {args.output.resolve()}")


if __name__ == "__main__":
    main()
