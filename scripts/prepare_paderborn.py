"""Audit every Paderborn signal and prepare a bearing-disjoint binary pilot."""

import csv
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from thesislib.common import ROOT, read_json, sha256, write_json
from thesislib.paderborn import (FILENAME, PAPER_URL, bearing_metadata, make_windows,
                                read_vibration, validate_protocol)


def main():
    config_path = ROOT / "configs/paderborn_baseline_v1.json"
    cfg = read_json(config_path)
    validate_protocol(cfg)
    summary_path = ROOT / "data/manifests/paderborn_v1_summary.json"
    if summary_path.exists():
        raise RuntimeError("Prepared protocol exists; preserve it. Use a new version for changed settings.")
    imported = read_json(ROOT / "data/manifests/paderborn_import.json")
    if imported["status"] != "complete" or imported["missing_official_archives"]:
        raise ValueError("Complete checked import is required.")
    raw_dir = Path(imported["extracted_directory"])
    out = ROOT / "data/processed/paderborn_v1"
    out.mkdir(parents=True, exist_ok=True)
    (out / "evaluator").mkdir(exist_ok=True)
    role_by_bearing = {bearing: role for role, bearings in cfg["bearing_groups"].items() for bearing in bearings}
    bins = {}
    rows, excluded = [], []
    counts = []
    files = sorted(raw_dir.glob("*/*.mat"))
    if len(files) != 2560:
        raise ValueError(f"Expected 2560 MAT recordings, found {len(files)}")
    for i, path in enumerate(files, 1):
        match = FILENAME.fullmatch(path.name)
        if not match or match[2] != path.parent.name:
            raise ValueError(f"Unexpected filename: {path}")
        condition, bearing, repetition = match[1], match[2], int(match[3])
        role = role_by_bearing[bearing]
        meta = bearing_metadata(bearing)
        selected = ((role in {"source_fit", "source_validation"} and condition == cfg["source_condition"])
                    or (role == "target_adaptation" and condition in cfg["target_conditions"])
                    or role == "evaluation")
        exclusion_reason = ""
        try:
            raw, quality = read_vibration(path)
        except (TypeError, ValueError, OSError) as error:
            exclusion_reason = f"{type(error).__name__}: {error}"
            excluded.append({"file": str(path.relative_to(ROOT)), "error": exclusion_reason,
                             "would_have_been_selected": selected})
            print(f"EXCLUDED {path.name}: {exclusion_reason}", flush=True)
            selected = False
            quality = {key: None for key in [
                "samples", "rms_recorded_units", "std_recorded_units", "embedded_time_span",
                "embedded_axis_mean_rate_if_seconds", "embedded_axis_rate_relative_difference_from_nominal",
                "nominal_fs_hz", "nominal_duration_s", "sampling_rate_policy"]}
        row = {"record_id": path.stem, "relative_path": str(path.relative_to(ROOT)),
               "sha256": sha256(path), "bearing_id": bearing, "condition": condition,
               "repetition": repetition, "bearing_group": role, "selected": selected,
               **meta, **quality, "windows": 0, "exclusion_reason": exclusion_reason}
        if selected:
            x = make_windows(raw, cfg)
            key = role if role.startswith("source_") else f"{role}_{condition}"
            bucket = bins.setdefault(key, {"x": [], "y": [], "ids": []})
            bucket["x"].append(x)
            bucket["y"].extend([meta["class_id"]] * len(x))
            bucket["ids"].extend(f"{path.stem}:{j * cfg['stride']}" for j in range(len(x)))
            row["windows"] = len(x)
        rows.append(row)
        if i % 80 == 0:
            print(f"Audited {i}/2560 recordings; completed bearing {bearing}", flush=True)
    if len(excluded) > .05 * len(files):
        raise RuntimeError("More than 5 percent invalid recordings; investigate before proceeding.")
    prepared_hashes = {}
    for key, bucket in sorted(bins.items()):
        x = np.concatenate(bucket["x"])
        y = np.asarray(bucket["y"], dtype=np.int64)
        ids = np.asarray(bucket["ids"])
        if key.startswith("source_"):
            destination = out / f"{key}.npz"
            np.savez_compressed(destination, x=x, y=y, ids=ids)
        else:
            destination = out / f"{key}_unlabeled.npz"
            np.savez_compressed(destination, x=x, ids=ids)
            labels = out / "evaluator" / f"{key}_labels.npz"
            np.savez_compressed(labels, y=y, ids=ids)
            prepared_hashes[str(labels.relative_to(out))] = sha256(labels)
        prepared_hashes[destination.name] = sha256(destination)
        selected_rows = [r for r in rows if r["selected"] and
                         (r["bearing_group"] if r["bearing_group"].startswith("source_") else
                          f"{r['bearing_group']}_{r['condition']}") == key]
        counts.append({"partition": key, "windows": len(y), "recordings": len(selected_rows),
                       "bearings": len({r["bearing_id"] for r in selected_rows}),
                       "healthy_windows": int((y == 0).sum()), "faulty_windows": int((y == 1).sum())})
        print(f"Saved {key}: {len(y)} windows from {len(selected_rows)} recordings", flush=True)
    records_path = ROOT / "data/manifests/paderborn_v1_records.csv"
    with records_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    discrepancy = [abs(r["embedded_axis_rate_relative_difference_from_nominal"]) for r in rows
                   if not r["exclusion_reason"]]
    write_json(summary_path, {
        "protocol_id": cfg["protocol_id"], "config_sha256": sha256(config_path),
        "raw_import_manifest_sha256": sha256(ROOT / "data/manifests/paderborn_import.json"),
        "record_manifest_sha256": sha256(records_path), "prepared_sha256": prepared_hashes,
        "raw_recordings_audit_attempted": len(rows), "readable_valid_recordings": len(rows) - len(excluded),
        "excluded_recordings": excluded,
        "selected_recordings": sum(r["selected"] for r in rows), "counts": counts,
        "maximum_abs_embedded_rate_difference_fraction": max(discrepancy),
        "records_over_1_percent_embedded_rate_difference": sum(d > .01 for d in discrepancy),
        "sampling_rate_evidence": PAPER_URL + " (section 4.1, page 7)",
        "label_evidence": PAPER_URL + " (tables 4, 5 and 7)",
        "limitations": [
            "Published nominal 64 kHz used despite embedded HostService timing discrepancies; physical frequency calibration unresolved.",
            "One fixed bearing split; only one healthy test bearing, so false-positive generalization is poorly estimated.",
            "Target evaluation changes both bearing identity and operating condition relative to source training.",
            "Windows and repetitions from the same bearing are correlated, not independent experiments.",
            "Binary healthy/faulty; not a reproduction of the earlier CWRU four-class task.",
            "Target adaptation pools are mixed-condition benchmarks, not healthy-only commissioning."
        ]})
    print(f"PREPARATION COMPLETE. Summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
