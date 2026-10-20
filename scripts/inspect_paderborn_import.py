"""Audit extracted filenames and read one vibration recording per bearing.

This is an import check, not a training or fault-detection performance result.
"""

from collections import Counter
import json
from pathlib import Path
import re

import numpy as np
from scipy.io import loadmat


ROOT = Path(__file__).resolve().parents[1]
PATTERN = re.compile(r"(N\d+_M\d+_F\d+)_(K[AIB]?\d+)_(\d+)\.mat")


def main():
    manifest = json.loads((ROOT / "data/manifests/paderborn_import.json").read_text())
    if manifest["status"] != "complete":
        raise RuntimeError("Wait until archive import is complete.")
    directory = Path(manifest["extracted_directory"])
    conditions = Counter()
    checked = []
    for archive in manifest["records"]:
        bearing = Path(archive["archive"]).stem
        paths = sorted((directory / bearing).rglob("*.mat"))
        repetitions = {}
        for path in paths:
            match = PATTERN.fullmatch(path.name)
            if not match or match[2] != bearing:
                raise RuntimeError(f"Unexpected filename: {path}")
            conditions[match[1]] += 1
            repetitions.setdefault(match[1], []).append(int(match[3]))
        if len(repetitions) != 4 or any(sorted(v) != list(range(1, 21)) for v in repetitions.values()):
            raise RuntimeError(f"Missing/duplicate repetitions or conditions: {bearing}")
        # Exactly one representative is read per bearing. Remaining files have
        # archive CRC, extracted size, and filename checks, not full signal QA.
        path = paths[0]
        record = loadmat(path, simplify_cells=True)[path.stem]
        vibration = [channel for channel in record["Y"] if channel["Name"] == "vibration_1"]
        if len(vibration) != 1:
            raise RuntimeError(f"Expected one vibration channel: {path}")
        channel = vibration[0]
        signal = np.asarray(channel["Data"], dtype=np.float64).reshape(-1)
        axis = np.asarray(record["X"][int(channel["XIndex"]) - 1]["Data"], dtype=np.float64).reshape(-1)
        if signal.size != axis.size or signal.size < 2:
            raise RuntimeError(f"Invalid time axis: {path}")
        if not np.isfinite(signal).all() or not np.isfinite(axis).all() or not np.all(np.diff(axis) > 0):
            raise RuntimeError(f"Invalid signal/time values: {path}")
        checked.append({"bearing_id": bearing, "file": str(path), "channel": "vibration_1",
                        "samples": signal.size, "time_axis_duration": float(axis[-1] - axis[0]),
                        "median_time_step": float(np.median(np.diff(axis))),
                        "standard_deviation": float(signal.std()),
                        "finite_values": True, "repetitions_per_condition": 20})
    result = {"status": "passed", "archive_count": len(checked),
              "mat_file_count": sum(conditions.values()), "conditions": dict(sorted(conditions.items())),
              "scope": "All archive CRCs/sizes/filenames; one MATLAB vibration recording per bearing read.",
              "not_checked": ["Full numerical QA of every signal", "Fault-label mapping",
                              "Model performance", "Physical units of embedded time axes"],
              "representative_recordings": checked}
    output = ROOT / "outputs/paderborn/import_inspection.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "representative_recordings"}, indent=2))
    print(f"Report: {output}")


if __name__ == "__main__":
    main()
