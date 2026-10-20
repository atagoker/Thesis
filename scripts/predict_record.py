"""Run a saved pilot classifier on an explicitly identified vibration channel.

This is a prototype inference command. Class probabilities are model outputs,
not validated industrial failure probabilities.
"""

import argparse
from fractions import Fraction
from pathlib import Path
import sys

import numpy as np
from scipy.io import loadmat
from scipy.signal import resample_poly
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from thesislib.common import ROOT, CLASSES, BearingCNN, normalize_windows, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", type=Path, required=True)
    parser.add_argument("--channel", required=True, help="Exact MATLAB channel key")
    parser.add_argument("--sampling-rate", type=int, required=True, help="Verified or explicitly assumed original Hz")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/demo/prediction.json")
    args = parser.parse_args()
    if args.sampling_rate <= 0:
        raise ValueError("Sampling rate must be positive.")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint["classes"] != CLASSES:
        raise ValueError("Unexpected label map.")
    cfg = checkpoint["config"]
    data = loadmat(args.file)
    if args.channel not in data:
        raise ValueError(f"Channel absent. Available non-metadata keys: {[k for k in data if not k.startswith('__')]}")
    signal = np.asarray(data[args.channel], dtype=np.float64).reshape(-1)
    if not np.isfinite(signal).all():
        raise ValueError("Non-finite signal.")
    ratio = Fraction(cfg["processed_sampling_rate_hz"], args.sampling_rate)
    signal = resample_poly(signal, ratio.numerator, ratio.denominator)
    trim = cfg["processed_edge_trim_samples"]
    signal = signal[trim:-trim]
    length = cfg["window_length"]
    windows = [signal[start:start+length] for start in range(0, len(signal)-length+1, cfg["stride"])]
    if not windows:
        raise ValueError("Recording is too short after preprocessing.")
    x = normalize_windows(np.stack(windows))[:, None, :]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BearingCNN().to(device).eval()
    model.load_state_dict(checkpoint["state_dict"])
    with torch.no_grad():
        p = torch.cat([model(torch.from_numpy(batch).to(device)).softmax(1).cpu()
                       for batch in np.array_split(x, max(1, int(np.ceil(len(x)/256))))]).numpy()
    mean_probability = p.mean(0)
    predicted = int(mean_probability.argmax())
    write_json(args.output, {
        "input_file": str(args.file.resolve()), "channel": args.channel,
        "sampling_rate_hz_supplied_by_user": args.sampling_rate,
        "checkpoint": str(args.checkpoint.resolve()), "checkpoint_role": checkpoint["role"],
        "window_count": len(x), "aggregation": "arithmetic mean of per-window probabilities",
        "predicted_record_class": CLASSES[predicted],
        "mean_class_probabilities": {name: float(value) for name, value in zip(CLASSES, mean_probability)},
        "per_window_predicted_class_counts": {name: int((p.argmax(1)==k).sum()) for k,name in enumerate(CLASSES)},
        "status": "prototype demonstration; no independent evaluation labels supplied",
    })
    print(f"Predicted recording class: {CLASSES[predicted]}", flush=True)
    print(f"Windows: {len(x)}; device: {device}; report: {args.output}", flush=True)


if __name__ == "__main__":
    main()
