"""Build time-blocked, nonoverlapping windows and label-free target archives."""

import csv
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from scipy.signal import resample_poly, welch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from thesislib.common import ROOT, CLASSES, normalize_windows, read_json, sha256, write_json


def region_bounds(n, gap):
    """Half a gap on either side of each raw split boundary."""
    first, second = int(n * 0.6), int(n * 0.8)
    half = gap // 2
    return {"fit": (0, first - half), "validation": (first + half, second - half),
            "test": (second + half, n)}


def main():
    config_path = ROOT / "configs/pilot_v1.json"
    cfg = read_json(config_path)
    catalog = read_json(ROOT / "data/manifests/cwru48_records.json")
    output = ROOT / "data/processed/pilot_v1"
    output.mkdir(parents=True, exist_ok=True)
    buckets = {(load, split): {"x": [], "y": [], "ids": []}
               for load in range(4) for split in ["fit", "validation", "test"]}
    window_rows, quality_rows = [], []
    examples = {}
    for rec in catalog["records"]:
        path = ROOT / rec["raw_path"]
        if sha256(path) != rec["sha256"]:
            raise RuntimeError(f"Raw checksum changed: {path}")
        raw = np.asarray(loadmat(path)[rec["channel_key"]], dtype=np.float64).reshape(-1)
        if rec["load_hp"] == 0 and (rec["class_id"] == 0 or rec["defect_size_inch"] == 0.007):
            examples[rec["class_id"]] = (resample_poly(raw, 1, 4), rec["name"])
        quality_rows.append({"record_id": rec["record_id"], "name": rec["name"],
                             "samples": len(raw), "duration_s_nominal": len(raw) / 48000,
                             "rms_raw": float(np.sqrt(np.mean(raw**2))),
                             "peak_raw": float(np.abs(raw).max()),
                             "rate_status": rec["sampling_rate_status"]})
        regions = region_bounds(len(raw), cfg["raw_boundary_gap_samples"])
        for split, (lo, hi) in regions.items():
            # Filter each raw region independently to avoid cross-partition support.
            processed = resample_poly(raw[lo:hi], 1, 4)
            trim = cfg["processed_edge_trim_samples"]
            processed = processed[trim:-trim]
            starts = range(0, len(processed) - cfg["window_length"] + 1, cfg["stride"])
            count = 0
            for start in starts:
                window = processed[start:start + cfg["window_length"]]
                if not np.isfinite(window).all() or window.std() < 1e-10:
                    raise ValueError(f"Invalid window in {rec['name']}")
                identifier = f"{rec['record_id']}:{split}:{start}"
                bucket = buckets[(rec["load_hp"], split)]
                bucket["x"].append(window)
                bucket["y"].append(rec["class_id"])
                bucket["ids"].append(identifier)
                raw_start = lo + (trim + start) * 4
                window_rows.append({
                    "window_id": identifier, "record_id": rec["record_id"],
                    "load_hp": rec["load_hp"], "class_id": rec["class_id"], "split": split,
                    "raw_start": raw_start, "raw_end_exclusive": raw_start + cfg["window_length"] * 4,
                    "raw_region_start": lo, "raw_region_end_exclusive": hi,
                })
                count += 1
            if not count:
                raise ValueError(f"No {split} windows in {rec['name']}; revise protocol before training.")
        print(f"Prepared {rec['name']}", flush=True)

    counts = []
    for (load, split), bucket in buckets.items():
        x = normalize_windows(np.stack(bucket["x"]))[:, None, :]
        y = np.asarray(bucket["y"], dtype=np.int64)
        ids = np.asarray(bucket["ids"])
        # Only source-load archives include labels in the training-facing folder.
        if load == cfg["source_load"]:
            np.savez_compressed(output / f"load{load}_{split}.npz", x=x, y=y, ids=ids)
        else:
            np.savez_compressed(output / f"load{load}_{split}_unlabeled.npz", x=x, ids=ids)
            # Curator/evaluator ground truth is separate, never read by adapt code.
            evaluator = output / "evaluator"
            evaluator.mkdir(exist_ok=True)
            np.savez_compressed(evaluator / f"load{load}_{split}_labels.npz", y=y, ids=ids)
        for class_id in range(4):
            counts.append({"load_hp": load, "split": split, "class_name": CLASSES[class_id],
                           "windows": int((y == class_id).sum())})
    manifest_path = ROOT / "data/manifests/pilot_v1_windows.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=window_rows[0].keys())
        writer.writeheader()
        writer.writerows(window_rows)
    write_json(ROOT / "data/manifests/pilot_v1_summary.json", {
        "protocol_id": cfg["protocol_id"], "config_sha256": sha256(config_path),
        "window_manifest_sha256": sha256(manifest_path), "counts": counts,
        "raw_file_count": len(catalog["records"]), "total_windows": len(window_rows),
        "shortest_record": min(quality_rows, key=lambda r: r["samples"]),
        "quality": quality_rows,
        "limitations": catalog["limitations"] + [
            "Temporal blocks share recordings and physical bearing identities.",
            "No independent-bearing or industrial deployment validation.",
            "Target pool includes mixed conditions; healthy-only commissioning is untested."],
    })
    figure_dir = ROOT / "outputs/pilot_v1/figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(4, 2, figsize=(12, 9), constrained_layout=True)
    for class_id, (signal, name) in sorted(examples.items()):
        segment = signal[:1200]
        axes[class_id, 0].plot(np.arange(len(segment)) / 12000, segment, lw=0.7, color="#117f89")
        axes[class_id, 0].set(title=f"{CLASSES[class_id]} ({name})", xlabel="Time (s)", ylabel="Recorded acceleration")
        f, p = welch(signal, fs=12000, nperseg=4096)
        axes[class_id, 1].semilogy(f, np.maximum(p, 1e-15), lw=0.8, color="#117f89")
        axes[class_id, 1].set(xlabel="Frequency (Hz)", ylabel="PSD (recorded units²/Hz)", xlim=(0, 6000))
    fig.suptitle("CWRU source-load examples: nominal 48 kHz to 12 kHz; raw amplitude preserved")
    fig.savefig(figure_dir / "real_signal_overview.png", dpi=170)
    plt.close(fig)
    print(f"Prepared {len(window_rows)} windows; manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
